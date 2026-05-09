"""
TRUST-AGENT Evaluation on VERITE Benchmark
===========================================
Evaluates TRUST-AGENT on the VERITE dataset.

VERITE has 3 classes:
  - true          → PRISTINE in our system
  - out-of-context → OUT-OF-CONTEXT in our system
  - miscaptioned  → OUT-OF-CONTEXT in our system (wrong caption = OOC)

Setup:
  1. git clone https://github.com/stevejpapad/image-text-verification
  2. cd image-text-verification
  3. python -c "from prepare_datasets import prepare_VERITE; prepare_VERITE(download_images=True)"
  4. cd ..
  5. python verite_eval.py

Usage:
  python verite_eval.py                          # all available samples
  python verite_eval.py --limit 100              # first 100 samples
  python verite_eval.py --limit 50 --ooc-only    # only OOC vs PRISTINE (skip miscaptioned)
  python verite_eval.py --skip-done              # resume interrupted run
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
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


# ── Load VERITE dataset ───────────────────────────────────────────────────────

def load_verite(
    verite_csv: str,
    limit: int | None,
    ooc_only: bool,
) -> List[Dict]:
    """
    Load VERITE.csv and convert labels to TRUST-AGENT format.

    VERITE labels    → TRUST-AGENT labels
    'true'           → PRISTINE
    'out-of-context' → OUT-OF-CONTEXT
    'miscaptioned'   → OUT-OF-CONTEXT
    """
    try:
        import pandas as pd
    except ImportError:
        LOG.error("pandas not installed. Run: pip install pandas")
        sys.exit(1)

    path = Path(verite_csv)
    if not path.exists():
        LOG.error("VERITE.csv not found at: %s", verite_csv)
        LOG.error("Run the download step first:")
        LOG.error("  cd image-text-verification")
        LOG.error("  python -c \"from prepare_datasets import prepare_VERITE; prepare_VERITE(download_images=True)\"")
        sys.exit(1)

    df = pd.read_csv(path)
    LOG.info("VERITE loaded: %d rows", len(df))
    LOG.info("Label distribution:\n%s", df['label'].value_counts().to_string())

    samples = []
    for i, row in df.iterrows():
        label = str(row.get('label', '')).strip().lower()
        caption = str(row.get('caption', '')).strip()
        img_path = str(row.get('image_path', '')).strip()

        # Skip miscaptioned if ooc_only mode
        if ooc_only and label == 'miscaptioned':
            continue

        # Convert label
        if label == 'true':
            ground_truth = 'PRISTINE'
        elif label in ('out-of-context', 'miscaptioned'):
            ground_truth = 'OUT-OF-CONTEXT'
        else:
            LOG.warning("Unknown label '%s' at row %d — skipping", label, i)
            continue

        if not caption or not img_path:
            continue

        # Fix image path — make it relative to project root
        full_img_path = img_path
        if not Path(img_path).is_absolute():
            # Try relative to VERITE folder
            verite_dir = path.parent
            candidate = verite_dir / img_path
            if candidate.exists():
                full_img_path = str(candidate)
            else:
                # Try with image-text-verification prefix
                candidate2 = Path("image-text-verification") / "VERITE" / img_path
                if candidate2.exists():
                    full_img_path = str(candidate2)

        if not Path(full_img_path).exists():
            LOG.debug("Image not found: %s — skipping", full_img_path)
            continue

        samples.append({
            "id":           str(i),
            "image_path":   full_img_path,
            "claim":        caption,
            "ground_truth": ground_truth,
            "verite_label": label,
            "source":       "VERITE",
        })

    LOG.info("Valid samples with images: %d", len(samples))

    if limit:
        # Take balanced subset
        pristine   = [s for s in samples if s['ground_truth'] == 'PRISTINE']
        ooc        = [s for s in samples if s['ground_truth'] == 'OUT-OF-CONTEXT']
        half       = limit // 2
        samples    = pristine[:half] + ooc[:half]
        LOG.info("Balanced to %d samples (%d PRISTINE + %d OOC)",
                 len(samples), len(pristine[:half]), len(ooc[:half]))

    return samples


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(results: List[Dict]) -> Dict:
    tp = tn = fp = fn = 0
    for r in results:
        p, t = r.get("predicted", ""), r.get("ground_truth", "")
        if   p == "OUT-OF-CONTEXT" and t == "OUT-OF-CONTEXT": tp += 1
        elif p == "PRISTINE"       and t == "PRISTINE":       tn += 1
        elif p == "OUT-OF-CONTEXT" and t == "PRISTINE":       fp += 1
        elif p == "PRISTINE"       and t == "OUT-OF-CONTEXT": fn += 1

    total     = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total else 0
    precision = tp / (tp + fp)    if (tp + fp) else 0
    recall    = tp / (tp + fn)    if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0)
    return dict(
        total=total, tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
    )


def compute_per_label(results: List[Dict]) -> Dict:
    """Break down accuracy by original VERITE label."""
    labels = {}
    for r in results:
        lbl = r.get("verite_label", "unknown")
        labels.setdefault(lbl, {"correct": 0, "total": 0})
        labels[lbl]["total"] += 1
        if r.get("predicted") == r.get("ground_truth"):
            labels[lbl]["correct"] += 1
    return {
        lbl: round(v["correct"] / v["total"], 3) if v["total"] else 0
        for lbl, v in labels.items()
    }


def print_report(metrics: Dict, results: List[Dict]) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("TRUST-AGENT — VERITE Benchmark Results")
    print(f"Samples: {metrics['total']}")
    print(sep)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision : {metrics['precision']:.4f}  ({metrics['precision']*100:.1f}%)")
    print(f"  Recall    : {metrics['recall']:.4f}  ({metrics['recall']*100:.1f}%)")
    print(f"  F1-Score  : {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.1f}%)")
    print()
    print("  Confusion Matrix:")
    print(f"    TP (OOC correctly detected)  : {metrics['tp']}")
    print(f"    TN (PRISTINE correctly ok)   : {metrics['tn']}")
    print(f"    FP (PRISTINE wrongly flagged): {metrics['fp']}")
    print(f"    FN (OOC missed)              : {metrics['fn']}")
    print()

    per_label = compute_per_label(results)
    if per_label:
        print("  Per-label accuracy:")
        for lbl, acc in per_label.items():
            print(f"    {lbl:<20} : {acc:.3f} ({acc*100:.1f}%)")
    print()

    print("COMPARISON WITH PUBLISHED SYSTEMS (VERITE benchmark):")
    print(f"  {'System':<35} {'Accuracy':>9}  Notes")
    print(f"  {'-'*35} {'-'*9}  {'-'*25}")
    rows = [
        ("CLIP baseline (zero-shot)",    "~60.0%", "CLIP similarity only"),
        ("DT-Transformer (VERITE paper)","~71.0%", "Trained on CHASMA"),
        ("SNIFFER (Qi, CVPR 2024)",      "~75.0%", "Fine-tuned LVLM"),
        ("HiEAG (2025)",                 "~80.0%", "Evidence-augmented"),
        ("TRUST-AGENT (ours)",
         f"{metrics['accuracy']*100:.1f}%",
         "Zero-shot, no training, CPU"),
    ]
    for name, acc, note in rows:
        print(f"  {name:<35} {acc:>9}  {note}")
    print(sep)

    wrong = [r for r in results if r.get("predicted") != r.get("ground_truth")]
    if wrong:
        print(f"\nWRONG PREDICTIONS ({len(wrong)}):")
        for r in wrong[:8]:
            print(f"  [{r.get('verite_label','?'):<15}] "
                  f"True={r['ground_truth']:<15} "
                  f"Pred={r.get('predicted','?'):<15} "
                  f"{r.get('claim','')[:50]}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TRUST-AGENT on VERITE benchmark"
    )
    parser.add_argument(
        "--verite-csv",
        default="image-text-verification/VERITE/VERITE.csv",
        help="Path to VERITE.csv (created after downloading images)"
    )
    parser.add_argument("--limit",     type=int, default=None,
                        help="Max samples (balanced). Default: all available")
    parser.add_argument("--output",    default="results/verite_results.csv")
    parser.add_argument("--ooc-only",  action="store_true",
                        help="Skip miscaptioned — only true vs out-of-context")
    parser.add_argument("--skip-done", action="store_true",
                        help="Resume interrupted run")
    args = parser.parse_args()

    # ── Load dataset ──────────────────────────────────────────────────────────
    samples = load_verite(
        verite_csv=args.verite_csv,
        limit=args.limit,
        ooc_only=args.ooc_only,
    )
    if not samples:
        LOG.error("No samples found. Check VERITE.csv path and image downloads.")
        sys.exit(1)

    LOG.info("Ready to evaluate %d samples", len(samples))

    # ── Load already-done IDs ─────────────────────────────────────────────────
    output_path = Path(args.output)
    done_ids: set  = set()
    existing: List = []
    if args.skip_done and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["id"])
                existing.append(row)
        LOG.info("Resuming — skipping %d already-done samples", len(done_ids))

    # ── Pipeline ──────────────────────────────────────────────────────────────
    config = Config()
    if not config.openai_api_key and not config.groq_api_key:
        LOG.error("Set OPENAI_API_KEY or GROQ_API_KEY in .env")
        sys.exit(1)

    orchestrator = TrustAgentOrchestrator(config)
    LOG.info("Pipeline ready. Starting VERITE evaluation...")

    new_results: List[Dict] = []
    total = len(samples)

    for i, sample in enumerate(samples):
        sid = sample["id"]
        if sid in done_ids:
            continue

        LOG.info(
            "[%d/%d] ID=%-6s | VERITE=%-15s | TRUST=%s",
            i + 1, total, sid,
            sample["verite_label"],
            sample["ground_truth"],
        )
        LOG.info("  Claim: %s", sample["claim"][:80])

        t0 = time.time()
        try:
            result = orchestrator.run(
                image_path=sample["image_path"],
                claim=sample["claim"],
            )
            elapsed = time.time() - t0
            correct = result.verdict == sample["ground_truth"]

            row = {
                "id":               sid,
                "verite_label":     sample["verite_label"],
                "ground_truth":     sample["ground_truth"],
                "predicted":        result.verdict,
                "correct":          "YES" if correct else "NO",
                "confidence":       result.confidence_percent,
                "entity_score":     round(result.entity_score, 3),
                "temporal_score":   round(result.temporal_score, 3),
                "credibility_score": round(result.credibility_score, 3),
                "final_score":      round(result.final_score, 3),
                "claim":            sample["claim"][:120],
                "time_sec":         round(elapsed, 1),
                "source":           "VERITE",
            }
            new_results.append(row)
            LOG.info(
                "  → %s (%d%%) [%s] in %.1fs",
                result.verdict, result.confidence_percent,
                "CORRECT" if correct else "WRONG", elapsed,
            )

        except Exception as exc:
            elapsed = time.time() - t0
            LOG.error("  Pipeline error: %s", exc)
            new_results.append({
                "id": sid, "verite_label": sample["verite_label"],
                "ground_truth": sample["ground_truth"],
                "predicted": "ERROR", "correct": "NO",
                "confidence": 0, "entity_score": 0, "temporal_score": 0,
                "credibility_score": 0, "final_score": 0,
                "claim": sample["claim"][:120],
                "time_sec": round(elapsed, 1), "source": "VERITE",
            })

        # Save after every sample — never lose progress
        all_results = existing + new_results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if all_results:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
                writer.writeheader()
                writer.writerows(all_results)

        time.sleep(0.5)

    # ── Final metrics ─────────────────────────────────────────────────────────
    all_results = existing + new_results
    valid = [r for r in all_results if r.get("predicted") not in ("ERROR", "")]

    if valid:
        metrics = compute_metrics(valid)
        print_report(metrics, valid)

        summary_path = output_path.parent / "verite_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("TRUST-AGENT — VERITE Benchmark\n")
            f.write(f"Samples  : {metrics['total']}\n")
            f.write(f"Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall   : {metrics['recall']:.4f}\n")
            f.write(f"F1-Score : {metrics['f1_score']:.4f}\n")
        LOG.info("Results : %s", output_path)
        LOG.info("Summary : %s", summary_path)
    else:
        LOG.error("No valid results.")


if __name__ == "__main__":
    main()