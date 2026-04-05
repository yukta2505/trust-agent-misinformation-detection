"""
TRUST-AGENT Evaluation Script
==============================
Runs the full pipeline on every sample in test_dataset.csv and
computes Accuracy, Precision, Recall, F1-score.

Usage:
    python evaluate.py
    python evaluate.py --dataset dataset/test_dataset.csv
    python evaluate.py --dataset dataset/test_dataset.csv --output results/eval_results.csv
    python evaluate.py --limit 10          # test only first 10 samples
    python evaluate.py --skip-done         # skip already-evaluated rows

Output:
    - results/eval_results.csv     (per-sample predictions + scores)
    - results/eval_summary.txt     (metrics table)
    - results/confusion_matrix.txt (TP/TN/FP/FN breakdown)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger(__name__)


# ── Import pipeline ────────────────────────────────────────────────────────────
try:
    from orchestrator import TrustAgentOrchestrator
    from backend.config import Config
except ImportError as e:
    print(f"ERROR: Could not import pipeline: {e}")
    print("Make sure you run this from inside trust-agent-misinformation-detection/")
    sys.exit(1)


# ── Metrics calculation ────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    """Compute Accuracy, Precision, Recall, F1 from results list."""
    tp = tn = fp = fn = 0

    for r in results:
        pred = r["predicted"]
        true = r["ground_truth"]
        if pred == "OUT-OF-CONTEXT" and true == "OUT-OF-CONTEXT":
            tp += 1
        elif pred == "PRISTINE" and true == "PRISTINE":
            tn += 1
        elif pred == "OUT-OF-CONTEXT" and true == "PRISTINE":
            fp += 1
        elif pred == "PRISTINE" and true == "OUT-OF-CONTEXT":
            fn += 1

    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp)    if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn)    if (tp + fn) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    return {
        "total": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy":  round(accuracy,  4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1_score":  round(f1,        4),
    }


def compute_per_type_metrics(results: list[dict]) -> dict:
    """Compute metrics broken down by sample type (time_trick, place_trick, etc.)"""
    types = {}
    for r in results:
        t = r.get("type", "unknown")
        if t not in types:
            types[t] = []
        types[t].append(r)

    breakdown = {}
    for t, rows in types.items():
        m = compute_metrics(rows)
        breakdown[t] = m
    return breakdown


# ── Report generation ──────────────────────────────────────────────────────────

def print_and_save_summary(metrics: dict, per_type: dict,
                            results: list[dict], output_dir: Path) -> None:
    """Print metrics to terminal and save to file."""

    lines = []
    lines.append("=" * 60)
    lines.append("TRUST-AGENT EVALUATION RESULTS")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Samples evaluated: {metrics['total']}")
    lines.append("=" * 60)
    lines.append("")
    lines.append("OVERALL METRICS")
    lines.append("-" * 40)
    lines.append(f"  Accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.1f}%)")
    lines.append(f"  Precision : {metrics['precision']:.4f}  ({metrics['precision']*100:.1f}%)")
    lines.append(f"  Recall    : {metrics['recall']:.4f}  ({metrics['recall']*100:.1f}%)")
    lines.append(f"  F1-Score  : {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.1f}%)")
    lines.append("")
    lines.append("CONFUSION MATRIX")
    lines.append("-" * 40)
    lines.append(f"  True Positives  (OOC correctly flagged) : {metrics['tp']}")
    lines.append(f"  True Negatives  (PRISTINE correctly ok) : {metrics['tn']}")
    lines.append(f"  False Positives (PRISTINE wrongly flagged): {metrics['fp']}")
    lines.append(f"  False Negatives (OOC missed)            : {metrics['fn']}")
    lines.append("")

    if per_type:
        lines.append("METRICS BY SAMPLE TYPE")
        lines.append("-" * 40)
        for t, m in per_type.items():
            lines.append(f"  {t:<20} | n={m['total']:>2} | "
                         f"Acc={m['accuracy']:.2f} | "
                         f"F1={m['f1_score']:.2f}")
        lines.append("")

    # Wrong predictions
    wrong = [r for r in results if r["predicted"] != r["ground_truth"]]
    if wrong:
        lines.append(f"WRONG PREDICTIONS ({len(wrong)} cases)")
        lines.append("-" * 40)
        for r in wrong:
            lines.append(f"  ID {r['id']:>3} | True={r['ground_truth']:<15} "
                         f"Pred={r['predicted']:<15} | {r['claim'][:60]}")
        lines.append("")

    # Comparison with literature
    lines.append("COMPARISON WITH EXISTING SYSTEMS")
    lines.append("-" * 40)
    lines.append(f"  COSMOS (Aneja 2020)        | ~73.0% accuracy | trained model")
    lines.append(f"  NewsCLIPpings (Abdel 2022) | ~78.0% accuracy | fine-tuned CLIP")
    lines.append(f"  EXCLAIM (Wu 2025)          | ~85.0% accuracy | agentic retrieval")
    lines.append(f"  TRUST-AGENT (ours)         | "
                 f"{metrics['accuracy']*100:.1f}% accuracy | zero-shot, no training")
    lines.append("")
    lines.append("NOTE: Published systems trained and tested on same distribution.")
    lines.append("TRUST-AGENT is zero-shot — no training data used at all.")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "eval_summary.txt"
    summary_path.write_text(report, encoding="utf-8")
    LOG.info("Summary saved to %s", summary_path)


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_evaluation(
    dataset_path: str,
    output_dir: Path,
    limit: int | None = None,
    skip_done: bool = False,
) -> None:

    config = Config()
    if not config.openai_api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    orchestrator = TrustAgentOrchestrator(config)
    LOG.info("Orchestrator ready. Starting evaluation...")

    # Load dataset
    samples = []
    with open(dataset_path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip placeholder rows
            caption = row.get("caption", "")
            if "[REAL CAPTION" in caption or "[REAL IMAGE" in caption:
                LOG.warning("Skipping row %s — placeholder caption not filled in",
                            row.get("id"))
                continue
            # Skip rows missing image
            img_path = row.get("image_path", "")
            if not Path(img_path).exists():
                LOG.warning("Skipping row %s — image not found: %s",
                            row.get("id"), img_path)
                continue
            samples.append(row)

    if limit:
        samples = samples[:limit]

    LOG.info("Loaded %d valid samples to evaluate", len(samples))

    # Load existing results if skip_done
    done_ids: set[str] = set()
    results_path = output_dir / "eval_results.csv"
    existing_results: list[dict] = []
    if skip_done and results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done_ids.add(row["id"])
                existing_results.append(row)
        LOG.info("Found %d already-evaluated samples, skipping them", len(done_ids))

    # Run pipeline on each sample
    new_results: list[dict] = []
    total = len(samples)

    for i, sample in enumerate(samples):
        sample_id = sample.get("id", str(i+1))

        if skip_done and sample_id in done_ids:
            continue

        claim    = sample.get("caption", sample.get("claim", "")).strip()
        img_path = sample.get("image_path", "").strip()
        truth    = sample.get("ground_truth", "").strip().upper()
        s_type   = sample.get("type", "unknown").strip()
        collector = sample.get("collector", "").strip()

        LOG.info("[%d/%d] ID=%s | %s | %s", i+1, total, sample_id, truth, s_type)
        LOG.info("  Claim: %s", claim[:80])

        start = time.time()
        try:
            result = orchestrator.run(image_path=img_path, claim=claim)
            elapsed = time.time() - start

            row = {
                "id":              sample_id,
                "image_path":      img_path,
                "claim":           claim[:100],
                "ground_truth":    truth,
                "predicted":       result.verdict,
                "correct":         "YES" if result.verdict == truth else "NO",
                "confidence":      result.confidence_percent,
                "entity_score":    round(result.entity_score, 3),
                "temporal_score":  round(result.temporal_score, 3),
                "credibility_score": round(result.credibility_score, 3),
                "final_score":     round(result.final_score, 3),
                "caption":         result.caption[:100],
                "explanation":     result.explanation[:150],
                "type":            s_type,
                "collector":       collector,
                "time_sec":        round(elapsed, 1),
                "errors":          "; ".join(result.errors),
            }
            new_results.append(row)

            status = "CORRECT" if result.verdict == truth else "WRONG"
            LOG.info("  → Predicted: %s (%s%%) [%s] in %.1fs",
                     result.verdict, result.confidence_percent, status, elapsed)

        except Exception as exc:
            LOG.error("  Pipeline error on ID=%s: %s", sample_id, exc)
            new_results.append({
                "id": sample_id, "image_path": img_path,
                "claim": claim[:100], "ground_truth": truth,
                "predicted": "ERROR", "correct": "NO",
                "confidence": 0, "entity_score": 0,
                "temporal_score": 0, "credibility_score": 0,
                "final_score": 0, "caption": "", "explanation": str(exc)[:150],
                "type": s_type, "collector": collector,
                "time_sec": round(time.time() - start, 1), "errors": str(exc),
            })

        # Save after every sample (so progress isn't lost)
        all_results = existing_results + new_results
        output_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = list(new_results[0].keys()) if new_results else []
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        # Small delay to avoid API rate limits
        time.sleep(1)

    # Compute and display metrics
    all_results = existing_results + new_results
    # Filter out errors for metrics
    valid = [r for r in all_results if r.get("predicted") not in ("ERROR", "")]
    if not valid:
        LOG.error("No valid results to compute metrics from.")
        return

    metrics  = compute_metrics(valid)
    per_type = compute_per_type_metrics(valid)
    print_and_save_summary(metrics, per_type, valid, output_dir)

    LOG.info("Results CSV: %s", results_path)
    LOG.info("Done. Evaluated %d samples.", len(valid))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TRUST-AGENT on a test dataset CSV."
    )
    parser.add_argument(
        "--dataset", default="dataset/test_dataset.csv",
        help="Path to test_dataset.csv"
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory for results (default: results/)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only first N samples (for quick testing)"
    )
    parser.add_argument(
        "--skip-done", action="store_true",
        help="Skip samples already in eval_results.csv (resume interrupted run)"
    )
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        output_dir=Path(args.output),
        limit=args.limit,
        skip_done=args.skip_done,
    )


if __name__ == "__main__":
    main()