"""
Standalone VERITE image downloader
====================================
Downloads VERITE images without needing imblearn, CLIP or any
of the other heavy dependencies from the original repo.

Run from inside image-text-verification/ folder:
    python download_verite.py

Or from project root:
    python download_verite.py --csv image-text-verification/VERITE/VERITE_articles.csv
                               --out image-text-verification/VERITE/images
                               --verite-out image-text-verification/VERITE/VERITE.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOG = logging.getLogger(__name__)


def download_image(url: str, dest: Path, timeout: int = 20) -> bool:
    """Download a single image. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 500:
        return True
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (trust-agent-research/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 500:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception as exc:
        LOG.debug("Failed %s: %s", url, exc)
        return False


def build_verite_csv(
    articles_csv: Path,
    images_dir: Path,
    output_csv: Path,
) -> None:
    """
    Read VERITE_articles.csv, download images, build VERITE.csv.

    VERITE_articles.csv columns:
      id, true_url, false_caption, true_caption, false_url, query, snopes_url

    Each article produces 3 rows in VERITE.csv:
      1. true image    + true caption    → label: true
      2. true image    + false caption   → label: miscaptioned
      3. false image   + true caption    → label: out-of-context
    """
    if not articles_csv.exists():
        LOG.error("VERITE_articles.csv not found at: %s", articles_csv)
        LOG.error("Make sure you cloned: git clone https://github.com/stevejpapad/image-text-verification")
        return

    images_dir.mkdir(parents=True, exist_ok=True)

    rows_out = []
    failed = 0
    total = 0

    with open(articles_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        articles = list(reader)

    LOG.info("Processing %d articles...", len(articles))

    for i, art in enumerate(articles):
        art_id     = art.get("id", str(i))
        true_url   = art.get("true_url", "").strip()
        false_url  = art.get("false_url", "").strip()
        true_cap   = art.get("true_caption", "").strip()
        false_cap  = art.get("false_caption", "").strip()

        if not true_cap:
            continue

        true_img_path  = images_dir / f"true_{art_id}.jpg"
        false_img_path = images_dir / f"false_{art_id}.jpg"

        # Download true image
        true_ok = download_image(true_url, true_img_path) if true_url else False
        # Download false image
        false_ok = download_image(false_url, false_img_path) if false_url else False

        total += 1
        if not true_ok and not false_ok:
            failed += 1
            LOG.warning("[%d/%d] Both images failed for id=%s", i+1, len(articles), art_id)
            continue

        # Row 1: true image + true caption → PRISTINE
        if true_ok and true_cap:
            rows_out.append({
                "caption":    true_cap,
                "image_path": str(true_img_path),
                "label":      "true",
            })

        # Row 2: true image + false caption → MISCAPTIONED (= OOC in binary)
        if true_ok and false_cap:
            rows_out.append({
                "caption":    false_cap,
                "image_path": str(true_img_path),
                "label":      "miscaptioned",
            })

        # Row 3: false image + true caption → OUT-OF-CONTEXT
        if false_ok and true_cap:
            rows_out.append({
                "caption":    true_cap,
                "image_path": str(false_img_path),
                "label":      "out-of-context",
            })

        if (i + 1) % 20 == 0:
            LOG.info(
                "[%d/%d] Downloaded so far: %d rows | %d failed",
                i+1, len(articles), len(rows_out), failed
            )

        time.sleep(0.1)  # gentle on servers

    # Write VERITE.csv
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["caption", "image_path", "label"])
        writer.writeheader()
        writer.writerows(rows_out)

    # Summary
    from collections import Counter
    counts = Counter(r["label"] for r in rows_out)
    LOG.info("Done!")
    LOG.info("VERITE.csv written: %s", output_csv)
    LOG.info("Total rows: %d", len(rows_out))
    LOG.info("  true           : %d", counts.get("true", 0))
    LOG.info("  miscaptioned   : %d", counts.get("miscaptioned", 0))
    LOG.info("  out-of-context : %d", counts.get("out-of-context", 0))
    LOG.info("Failed articles  : %d / %d", failed, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VERITE images")
    parser.add_argument(
        "--csv",
        default="VERITE/VERITE_articles.csv",
        help="Path to VERITE_articles.csv"
    )
    parser.add_argument(
        "--out",
        default="VERITE/images",
        help="Directory to save downloaded images"
    )
    parser.add_argument(
        "--verite-out",
        default="VERITE/VERITE.csv",
        help="Output path for final VERITE.csv"
    )
    args = parser.parse_args()

    build_verite_csv(
        articles_csv=Path(args.csv),
        images_dir=Path(args.out),
        output_csv=Path(args.verite_out),
    )


if __name__ == "__main__":
    main()