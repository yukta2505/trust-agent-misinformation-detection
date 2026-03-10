from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.newsclippings_loader import HF_BASE_URL, build_example, load_split


def _download_file(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _candidate_urls(image_path: str, base_url: str) -> list[str]:
    path = image_path.lstrip("/")
    candidates = [
        f"{base_url.rstrip('/')}/{path}",
    ]
    if path.startswith("visual_news/"):
        candidates.append(f"{base_url.rstrip('/')}/{path[len('visual_news/'):]}")
    if path.startswith("visual_news/origin/"):
        candidates.append(f"{base_url.rstrip('/')}/{path[len('visual_news/origin/'):]}")
    return candidates


def build_sample(
    split: str,
    limit: int,
    output_dir: Path,
    jsonl_path: Path,
    image_base_url: str,
) -> None:
    annotations, metadata = load_split(split)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    attempts = 0
    seen_paths: set[str] = set()
    max_attempts = max(limit * 50, limit)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for annotation in annotations:
            example = build_example(annotation=annotation, metadata=metadata)
            image_rel_path = metadata.get(str(example.id), {}).get("image_path")
            if not image_rel_path:
                continue
            if image_rel_path in seen_paths:
                continue
            seen_paths.add(image_rel_path)
            attempts += 1
            if attempts > max_attempts:
                break
            image_name = f"{example.id}{Path(image_rel_path).suffix or '.jpg'}"
            local_image_path = output_dir / "images" / image_name
            if not local_image_path.exists():
                downloaded = False
                last_error: Exception | None = None
                for image_url in _candidate_urls(image_rel_path, image_base_url):
                    try:
                        _download_file(image_url, local_image_path)
                        downloaded = True
                        break
                    except requests.HTTPError as exc:
                        last_error = exc
                        print(f"Skipping {image_url} due to HTTP error: {exc}")
                    except requests.RequestException as exc:
                        last_error = exc
                        print(f"Skipping {image_url} due to request error: {exc}")
                if not downloaded:
                    if last_error is not None:
                        print(f"No valid URL found for {image_rel_path}. Last error: {last_error}")
                    continue

            record: dict[str, Any] = asdict(example)
            record.update(
                {
                    "image_path": str(local_image_path),
                    "article_title": example.title,
                }
            )
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            written += 1
            if written >= limit:
                break

    if written < limit:
        raise RuntimeError(
            "Sample download incomplete. The split may have missing images or they may not be hosted. "
            f"Wrote {written} of {limit} after {attempts} attempts."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a small NewsCLIPpings sample with images and emit a JSONL dataset."
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output-dir", default="dataset/newsclippings_sample")
    parser.add_argument("--jsonl", default="dataset/newsclippings_sample/sample.jsonl")
    parser.add_argument(
        "--image-base-url",
        default=HF_BASE_URL,
        help="Base URL for VisualNews image hosting. Defaults to the NewsCLIPpings HF base URL.",
    )
    args = parser.parse_args()

    build_sample(
        split=args.split,
        limit=max(args.limit, 1),
        output_dir=Path(args.output_dir),
        jsonl_path=Path(args.jsonl),
        image_base_url=args.image_base_url,
    )


if __name__ == "__main__":
    main()
