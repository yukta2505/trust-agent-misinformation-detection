"""
NewsCLIPpings Benchmark Evaluator for TRUST-AGENT
===================================================
Downloads ONLY annotation JSONs from HuggingFace (few MB),
fetches images on-demand via VisualNews image_url field,
evaluates TRUST-AGENT on N samples.

NO 1TB DOWNLOAD REQUIRED.

How image fetching works (fixed):
  - VisualNews data.json from twelcone/VisualNews (HuggingFace dataset)
    contains a live `image_url` field for every record pointing to the
    original BBC / Guardian / WashPost / USAToday CDN.
  - We fetch using that direct URL with the `requests` library
    (better timeout control than urllib — no more hangs/crashes).
  - Falls back to constructing URL from image_path if image_url missing.

Usage:
    pip install huggingface_hub requests
    python benchmark_eval.py --dry              # download only, no pipeline
    python benchmark_eval.py --limit 50
    python benchmark_eval.py --limit 100 --skip-done
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── requests is required (better than urllib for timeout handling) ─────────────
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger(__name__)

# ── Import TRUST-AGENT ────────────────────────────────────────────────────────
try:
    from orchestrator import TrustAgentOrchestrator
    from backend.config import Config
except ImportError as e:
    print(f"ERROR: Cannot import pipeline: {e}")
    print("Run from trust-agent-misinformation-detection/ folder")
    sys.exit(1)


# ── Shared requests session with retry + timeout ──────────────────────────────

def _make_session() -> requests.Session:
    """
    Create a requests.Session with:
      - 3 retries on connection / 500 / 502 / 503 / 504 errors
      - 10s connect timeout, 20s read timeout (set per-call)
      - A browser-like User-Agent so CDNs don't block us
    """
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,            # wait 1s, 2s, 4s between retries
        status_forcelist={500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
    })
    return session


SESSION = _make_session()

# Exposed for warnings when image fetch fails (kept global for minimal plumbing).
_LAST_FETCH_ERROR: Optional[str] = None
_LAST_FETCH_URL: Optional[str] = None


# ── Step 1: Download annotations ──────────────────────────────────────────────

def download_annotations(cache_dir: Path) -> tuple:
    """
    Download NewsCLIPpings test split   → g-luo/news-clippings  (model repo)
    Download VisualNews metadata        → twelcone/VisualNews    (dataset repo)

    The VisualNews data.json contains an `image_url` field for every record
    pointing to the live CDN of the original news outlet (BBC, Guardian, etc.).
    This is the key that makes on-demand image fetching work.

    Returns (annotations list, vn_map dict  {str(id): record})
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        LOG.error("huggingface_hub not installed.  Run: pip install huggingface_hub")
        sys.exit(1)

    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 1a. NewsCLIPpings test annotations ───────────────────────────────────
    test_cache = cache_dir / "test.json"
    if not test_cache.exists():
        LOG.info("Downloading NewsCLIPpings test annotations …")
        try:
            path = hf_hub_download(
                repo_id="g-luo/news-clippings",
                filename="data/merged_balanced/test.json",
                repo_type="model",          # g-luo repo is a MODEL repo
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
            )
            import shutil
            shutil.copy(path, str(test_cache))
            LOG.info("Test annotations saved → %s", test_cache)
        except Exception as exc:
            LOG.error("Could not download test annotations: %s", exc)
            LOG.error("Manual fix:")
            LOG.error("  from huggingface_hub import hf_hub_download")
            LOG.error("  hf_hub_download(repo_id='g-luo/news-clippings',")
            LOG.error("    filename='news_clippings/data/merged_balanced/test.json',")
            LOG.error("    repo_type='model', local_dir='./cache')")
            sys.exit(1)
    else:
        LOG.info("Using cached test annotations: %s", test_cache)

    with open(test_cache, encoding="utf-8") as f:
        test_data = json.load(f)
    annotations = test_data.get("annotations", test_data)
    LOG.info("Test split: %d annotations", len(annotations))

    # ── 1b. VisualNews metadata (has image_url per record) ───────────────────
    # NOTE: avoid colliding with older caches that may contain unrelated
    # metadata (e.g. g-luo/news-clippings metadata/test.json without image_url).
    vn_cache = cache_dir / "visualnews_data_twelcone.json"
    legacy_cache = cache_dir / "visualnews_data.json"

    if not vn_cache.exists():
        if legacy_cache.exists():
            LOG.warning(
                "Found legacy VisualNews cache %s but will NOT use it. "
                "This script expects twelcone/VisualNews `data.json` "
                "which includes an `image_url` field.",
                legacy_cache,
            )

        LOG.info("Downloading VisualNews metadata from twelcone/VisualNews …")
        LOG.info("(~356 MB JSON — this takes a few minutes, downloads once)")
        try:
            path = hf_hub_download(
                repo_id="twelcone/VisualNews",
                filename="data.json",
                repo_type="dataset",        # twelcone repo is a DATASET repo
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
            )
            import shutil
            shutil.copy(path, str(vn_cache))
            LOG.info("VisualNews metadata saved → %s", vn_cache)
        except Exception as exc:
            LOG.warning("Could not download VisualNews metadata: %s", exc)
            LOG.warning("Will attempt image_path URL construction (may 404)")
            return annotations, {}
    else:
        LOG.info("Using cached VisualNews metadata: %s", vn_cache)

    LOG.info("Loading VisualNews metadata into memory …")
    with open(vn_cache, encoding="utf-8") as f:
        vn_raw = json.load(f)

    # Normalise to  {str(id): record}  regardless of source format
    if isinstance(vn_raw, list):
        vn_map: Dict[str, Any] = {str(item["id"]): item for item in vn_raw}
    elif isinstance(vn_raw, dict):
        vn_map = {str(k): v for k, v in vn_raw.items()}
    else:
        LOG.warning("Unexpected VisualNews format — treating as empty")
        vn_map = {}

    LOG.info("VisualNews metadata: %d entries loaded", len(vn_map))
    return annotations, vn_map


# ── Step 2: Fetch individual images ───────────────────────────────────────────

# Fallback CDN base (only used if image_url is absent from metadata)
_VN_IMAGE_BASE = (
    "https://huggingface.co/g-luo/news-clippings/resolve/main/visual_news/origin"
)


def fetch_image(
    image_path: str,
    dest: Path,
    vn_record: Optional[Dict] = None,
    timeout: tuple = (10, 25),
) -> bool:
    """
    Download a single image.

    Priority order for the URL:
      1. vn_record["image_url"]   — direct CDN link (BBC / Guardian / etc.)
      2. vn_record["image_path"]  — fallback: construct from path
      3. image_path argument       — fallback: construct from path
      4. HuggingFace CDN base      — last resort (usually 404)

    Args:
        image_path : relative path from VisualNews data.json
        dest       : where to save the file
        vn_record  : the matching VisualNews metadata record
        timeout    : (connect_timeout, read_timeout) in seconds
    """
    if dest.exists() and dest.stat().st_size > 500:
        return True                     # already cached

    dest.parent.mkdir(parents=True, exist_ok=True)

    global _LAST_FETCH_ERROR, _LAST_FETCH_URL
    _LAST_FETCH_ERROR = None
    _LAST_FETCH_URL = None

    candidates: List[str] = []

    # ── Priority 1: direct image_url from VisualNews ─────────────────────────
    if vn_record:
        direct = vn_record.get("image_url", "").strip()
        if direct:
            candidates.append(direct)

    # ── Priority 2 & 3: construct URL from path ───────────────────────────────
    for raw_path in [
        (vn_record or {}).get("image_path", ""),
        image_path,
    ]:
        if not raw_path:
            continue
        clean = raw_path.strip("/")
        for prefix in ("visual_news/origin/", "visual_news/", "origin/"):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        url = f"{_VN_IMAGE_BASE}/{clean}"
        if url not in candidates:
            candidates.append(url)

    if not candidates:
        LOG.debug("No URL candidates for dest=%s", dest.name)
        return False

    for url in candidates:
        _LAST_FETCH_URL = url
        try:
            resp = SESSION.get(url, timeout=timeout, stream=True)
            if resp.status_code == 200:
                data = resp.content
                if len(data) > 500:           # ignore tiny error pages
                    dest.write_bytes(data)
                    LOG.debug("Saved image → %s  (%d bytes)", dest.name, len(data))
                    return True
                else:
                    LOG.debug("Response too small (%d bytes) for %s", len(data), url)
                    _LAST_FETCH_ERROR = f"Response too small ({len(data)} bytes)"
            else:
                LOG.debug("HTTP %d for %s", resp.status_code, url)
                _LAST_FETCH_ERROR = f"HTTP {resp.status_code}"
        except requests.exceptions.Timeout:
            LOG.debug("Timeout fetching %s", url)
            _LAST_FETCH_ERROR = "Timeout"
        except requests.exceptions.ConnectionError as exc:
            LOG.debug("Connection error for %s: %s", url, exc)
            _LAST_FETCH_ERROR = f"ConnectionError: {exc}"
        except Exception as exc:
            LOG.debug("Unexpected error for %s: %s", url, exc)
            _LAST_FETCH_ERROR = f"{type(exc).__name__}: {exc}"

    return False


# ── Step 3: Build balanced sample list ────────────────────────────────────────

def build_samples(
    annotations: List[Dict],
    vn_map: Dict[str, Any],
    limit: int,
    image_dir: Path,
    fetch_images: bool = True,
    *,
    fetch_timeout: tuple = (10, 25),
) -> List[Dict[str, Any]]:
    """
    Build a balanced list of `limit` samples (50 % PRISTINE / 50 % OOC).
    Downloads only the images actually needed.
    """
    pristine_want  = limit // 2
    falsified_want = limit - pristine_want
    pristine: List[Dict] = []
    falsified: List[Dict] = []

    LOG.info(
        "Building %d balanced samples (%d PRISTINE + %d OOC) …",
        limit, pristine_want, falsified_want,
    )

    for ann in annotations:
        if len(pristine) >= pristine_want and len(falsified) >= falsified_want:
            break

        ann_id = str(ann.get("id", ""))
        img_id = str(ann.get("image_id", ann_id))
        is_ooc = bool(ann.get("falsified", False))

        # Skip if we already have enough of this class
        if is_ooc  and len(falsified) >= falsified_want:
            continue
        if not is_ooc and len(pristine)  >= pristine_want:
            continue

        # ── Look up VisualNews record ─────────────────────────────────────────
        # NewsCLIPpings ann_id  → caption  (use ann_id record)
        # NewsCLIPpings img_id  → image    (use img_id record; may differ for OOC)
        vn_ann = vn_map.get(ann_id, {})
        vn_img = vn_map.get(img_id, vn_ann)    # img record (may be same)

        caption   = vn_ann.get("caption", "").strip()
        img_path  = vn_img.get("image_path", "").strip()

        if not caption:
            LOG.debug("Skipping ann_id=%s — no caption in VisualNews", ann_id)
            continue

        # ── Build base sample dict ────────────────────────────────────────────
        sample: Dict[str, Any] = {
            "id":           ann_id,
            "claim":        caption,
            "ground_truth": "OUT-OF-CONTEXT" if is_ooc else "PRISTINE",
            "source":       "NewsCLIPpings",
        }

        # ── Fetch image ───────────────────────────────────────────────────────
        if fetch_images:
            if not img_path and not vn_img.get("image_url"):
                LOG.debug("No image path/url for ann_id=%s — skipping", ann_id)
                continue

            suffix = Path(img_path).suffix if img_path else ".jpg"
            if not suffix:
                suffix = ".jpg"
            dest = image_dir / f"nc_{ann_id}_{img_id}{suffix}"

            LOG.info(
                "Fetching image for ann=%s img=%s OOC=%s …",
                ann_id, img_id, is_ooc,
            )
            ok = fetch_image(
                img_path,
                dest,
                vn_record=vn_img,
                timeout=fetch_timeout,
            )
            if not ok:
                detail = _LAST_FETCH_ERROR or "unknown error"
                if _LAST_FETCH_URL:
                    detail = f"{detail} @ {_LAST_FETCH_URL}"
                LOG.warning(
                    "Could not fetch image for ann_id=%s img_id=%s — skipping (%s)",
                    ann_id, img_id, detail,
                )
                continue
            sample["image_path"] = str(dest)
        else:
            # --dry mode: record a placeholder path (won't be used by pipeline)
            suffix = Path(img_path).suffix if img_path else ".jpg"
            sample["image_path"] = str(
                image_dir / f"nc_{ann_id}_{img_id}{suffix or '.jpg'}"
            )

        if is_ooc:
            falsified.append(sample)
        else:
            pristine.append(sample)

    samples = pristine + falsified
    LOG.info(
        "Sample build complete: %d total  (%d PRISTINE, %d OOC)",
        len(samples), len(pristine), len(falsified),
    )

    if len(samples) < limit:
        LOG.warning(
            "Only %d/%d samples available.  "
            "This usually means VisualNews image_url fields are absent or "
            "images are unreachable.  Check your internet connection.",
            len(samples), limit,
        )

    return samples


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(results: List[Dict]) -> Dict:
    tp = tn = fp = fn = 0
    for r in results:
        p = r.get("predicted", "")
        t = r.get("ground_truth", "")
        if   p == "OUT-OF-CONTEXT" and t == "OUT-OF-CONTEXT": tp += 1
        elif p == "PRISTINE"       and t == "PRISTINE":       tn += 1
        elif p == "OUT-OF-CONTEXT" and t == "PRISTINE":       fp += 1
        elif p == "PRISTINE"       and t == "OUT-OF-CONTEXT": fn += 1

    total     = tp + tn + fp + fn
    accuracy  = (tp + tn) / total         if total          else 0.0
    precision = tp / (tp + fp)            if (tp + fp)      else 0.0
    recall    = tp / (tp + fn)            if (tp + fn)      else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)

    return dict(
        total=total, tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
    )


def print_report(metrics: Dict, results: List[Dict]) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("TRUST-AGENT — NewsCLIPpings Benchmark Results")
    print(f"Samples evaluated : {metrics['total']}")
    print(sep)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision : {metrics['precision']:.4f}  ({metrics['precision']*100:.1f}%)")
    print(f"  Recall    : {metrics['recall']:.4f}  ({metrics['recall']*100:.1f}%)")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.1f}%)")
    print()
    print("  Confusion Matrix:")
    print(f"    TP (OOC correctly flagged)    : {metrics['tp']}")
    print(f"    TN (PRISTINE correctly ok)    : {metrics['tn']}")
    print(f"    FP (PRISTINE wrongly flagged) : {metrics['fp']}")
    print(f"    FN (OOC missed)               : {metrics['fn']}")
    print()
    print("COMPARISON WITH PUBLISHED SYSTEMS (NewsCLIPpings test split):")
    rows = [
        ("COSMOS (Aneja, CVPR 2020)",  "~73.0%", "Trained model, GPU required"),
        ("CCN (Abdelnabi, CVPR 2022)", "~77.8%", "Trained model, GPU required"),
        ("SNIFFER (Qi, CVPR 2024)",    "~88.4%", "Trained model, GPU required"),
        ("EXCLAIM (Wu, arXiv 2025)",   "~89.3%", "Zero-shot agentic, GPU for index"),
        ("E2LVLM (Wu, arXiv 2025)",    "90.34%", "Fine-tuned LVLM, GPU required"),
        (
            "TRUST-AGENT (ours)",
            f"{metrics['accuracy']*100:.1f}%",
            "Zero-shot, no training, CPU only",
        ),
    ]
    print(f"  {'System':<36} {'Accuracy':>9}  Notes")
    print(f"  {'-'*36} {'-'*9}  {'-'*30}")
    for name, acc, note in rows:
        print(f"  {name:<36} {acc:>9}  {note}")
    print(sep)

    wrong = [r for r in results if r.get("predicted") != r.get("ground_truth")]
    if wrong:
        print(f"\nWRONG PREDICTIONS ({len(wrong)}):")
        for r in wrong[:10]:
            print(
                f"  ID={r['id']:<12} "
                f"True={r['ground_truth']:<15} "
                f"Pred={r.get('predicted','?'):<15} "
                f"Score={float(r.get('final_score', 0)):.3f}  "
                f"{r.get('claim','')[:45]}"
            )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TRUST-AGENT on NewsCLIPpings benchmark"
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Total samples to evaluate (balanced 50/50 PRISTINE/OOC)",
    )
    parser.add_argument(
        "--output", default="results/benchmark_results.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--cache", default=".newsclippings_cache",
        help="Directory for cached downloads",
    )
    parser.add_argument(
        "--dry", action="store_true",
        help="Download data + fetch images only — do NOT run pipeline",
    )
    parser.add_argument(
        "--connect-timeout", type=float, default=10.0,
        help="Image fetch connect timeout (seconds)",
    )
    parser.add_argument(
        "--read-timeout", type=float, default=25.0,
        help="Image fetch read timeout (seconds)",
    )
    parser.add_argument(
        "--skip-done", action="store_true",
        help="Resume: skip sample IDs already present in output CSV",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    image_dir = cache_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # ── Download annotations + VisualNews metadata ────────────────────────────
    annotations, vn_map = download_annotations(cache_dir)

    if not vn_map:
        LOG.warning(
            "VisualNews metadata is empty.  Image fetching will likely fail.\n"
            "Make sure you have internet access and huggingface_hub installed."
        )

    # ── Build balanced sample list (fetches images) ───────────────────────────
    samples = build_samples(
        annotations,
        vn_map,
        args.limit,
        image_dir,
        fetch_images=True,              # always fetch; --dry stops before pipeline
        fetch_timeout=(args.connect_timeout, args.read_timeout),
    )

    if not samples:
        LOG.error(
            "No samples could be loaded.\n"
            "Possible causes:\n"
            "  1. No internet / HuggingFace blocked\n"
            "  2. VisualNews image_url fields are empty (check vn_map sample)\n"
            "  3. All CDN image URLs return non-200\n"
            "Try: python benchmark_eval.py --dry  to see vn_map sample output"
        )
        # Print a diagnostic sample so the user can investigate
        if vn_map:
            sample_id = next(iter(vn_map))
            LOG.info("Sample vn_map entry (id=%s): %s", sample_id, vn_map[sample_id])
        sys.exit(1)

    if args.dry:
        LOG.info("--dry flag set. Stopping before pipeline execution.")
        LOG.info("Sample 0: %s", samples[0])
        if vn_map:
            first_id = samples[0]["id"]
            LOG.info("VisualNews record for sample 0: %s", vn_map.get(first_id, "NOT FOUND"))
        return

    # ── Load already-completed IDs (resume support) ───────────────────────────
    output_path = Path(args.output)
    done_ids: set       = set()
    existing: List[Dict] = []

    if args.skip_done and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["id"])
                existing.append(row)
        LOG.info("Resuming — %d samples already done, skipping them", len(done_ids))

    # ── Initialise pipeline ───────────────────────────────────────────────────
    config = Config()
    if not getattr(config, "openai_api_key", None) and \
       not getattr(config, "groq_api_key", None):
        LOG.error(
            "No LLM API key found.\n"
            "Set OPENAI_API_KEY or GROQ_API_KEY in your .env file."
        )
        sys.exit(1)

    orchestrator = TrustAgentOrchestrator(config)
    LOG.info("Pipeline ready. Evaluating %d samples …", len(samples))

    new_results: List[Dict] = []
    total = len(samples)

    CSV_FIELDS = [
        "id", "ground_truth", "predicted", "correct",
        "confidence", "entity_score", "temporal_score",
        "credibility_score", "final_score",
        "claim", "time_sec", "source",
    ]

    for i, sample in enumerate(samples):
        sid = sample["id"]

        if sid in done_ids:
            LOG.info("[%d/%d] ID=%-12s  SKIPPED (already done)", i + 1, total, sid)
            continue

        LOG.info(
            "[%d/%d] ID=%-12s | %-15s | %s",
            i + 1, total, sid, sample["ground_truth"],
            sample["claim"][:70],
        )

        t0 = time.time()
        try:
            result  = orchestrator.run(
                image_path=sample["image_path"],
                claim=sample["claim"],
            )
            elapsed = time.time() - t0
            correct = result.verdict == sample["ground_truth"]

            row = {
                "id":                sid,
                "ground_truth":      sample["ground_truth"],
                "predicted":         result.verdict,
                "correct":           "YES" if correct else "NO",
                "confidence":        result.confidence_percent,
                "entity_score":      round(result.entity_score, 3),
                "temporal_score":    round(result.temporal_score, 3),
                "credibility_score": round(result.credibility_score, 3),
                "final_score":       round(result.final_score, 3),
                "claim":             sample["claim"][:120],
                "time_sec":          round(elapsed, 1),
                "source":            "NewsCLIPpings",
            }
            new_results.append(row)

            LOG.info(
                "  → %-16s (%3d%%)  [%s]  %.1fs",
                result.verdict, result.confidence_percent,
                "CORRECT" if correct else "WRONG", elapsed,
            )

        except Exception as exc:
            elapsed = time.time() - t0
            LOG.error("  Pipeline error for ID=%s: %s", sid, exc)
            new_results.append({
                "id":                sid,
                "ground_truth":      sample["ground_truth"],
                "predicted":         "ERROR",
                "correct":           "NO",
                "confidence":        0,
                "entity_score":      0,
                "temporal_score":    0,
                "credibility_score": 0,
                "final_score":       0,
                "claim":             sample["claim"][:120],
                "time_sec":          round(elapsed, 1),
                "source":            "NewsCLIPpings",
            })

        # ── Save progress after every sample (crash-safe) ────────────────────
        all_rows = existing + new_results
        if all_rows:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(all_rows)

        time.sleep(0.4)     # be gentle on APIs

    # ── Final report ──────────────────────────────────────────────────────────
    all_rows = existing + new_results
    valid    = [r for r in all_rows if r.get("predicted") not in ("ERROR", "")]

    if valid:
        metrics = compute_metrics(valid)
        print_report(metrics, valid)

        summary_path = output_path.parent / "benchmark_summary.txt"
        summary_path.write_text(
            "TRUST-AGENT — NewsCLIPpings Benchmark Summary\n"
            f"Samples   : {metrics['total']}\n"
            f"Accuracy  : {metrics['accuracy']:.4f}  "
            f"({metrics['accuracy']*100:.1f}%)\n"
            f"Precision : {metrics['precision']:.4f}\n"
            f"Recall    : {metrics['recall']:.4f}\n"
            f"F1-Score  : {metrics['f1_score']:.4f}\n",
            encoding="utf-8",
        )
        LOG.info("Results saved  → %s", output_path)
        LOG.info("Summary saved  → %s", summary_path)
    else:
        LOG.error(
            "No valid results to report. "
            "Check pipeline errors above — all predictions were ERROR."
        )


if __name__ == "__main__":
    main()
