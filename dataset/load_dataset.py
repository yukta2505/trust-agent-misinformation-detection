from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.newsclippings_loader import load_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Load NewsCLIPpings remotely without local dataset download.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=3, help="Number of examples to print.")
    args = parser.parse_args()

    examples = load_examples(split=args.split, limit=args.limit)
    serialized = [asdict(example) for example in examples]
    print(json.dumps(serialized, indent=2))


if __name__ == "__main__":
    main()
