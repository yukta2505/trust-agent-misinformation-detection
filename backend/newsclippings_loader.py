from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

HF_BASE_URL = "https://huggingface.co/g-luo/news-clippings/resolve/main"
VALID_SPLITS = {"train", "val", "test"}


@dataclass(frozen=True)
class NewsClippingsExample:
    id: int
    image_id: int
    falsified: bool
    source_dataset: int
    similarity_score: float
    caption: str
    timestamp: str
    source: str
    topic: str
    caption_entities_spacy: list[list[str]]
    title: str


def _fetch_json(url: str, timeout: int = 60) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_split(split: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split: {split}. Use one of {sorted(VALID_SPLITS)}.")

    annotations_url = f"{HF_BASE_URL}/data/merged_balanced/{split}.json"
    metadata_url = f"{HF_BASE_URL}/metadata/{split}.json"

    annotations_payload = _fetch_json(annotations_url)
    metadata_payload = _fetch_json(metadata_url)

    annotations = annotations_payload.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError("Unexpected annotations format.")
    if not isinstance(metadata_payload, dict):
        raise ValueError("Unexpected metadata format.")

    return annotations, metadata_payload


def build_example(annotation: dict[str, Any], metadata: dict[str, dict[str, Any]]) -> NewsClippingsExample:
    key = str(annotation["id"])
    item = metadata.get(key)
    if item is None:
        raise KeyError(f"Metadata missing for annotation id={annotation['id']}.")

    return NewsClippingsExample(
        id=int(annotation["id"]),
        image_id=int(annotation["image_id"]),
        falsified=bool(annotation["falsified"]),
        source_dataset=int(annotation["source_dataset"]),
        similarity_score=float(annotation["similarity_score"]),
        caption=str(item.get("caption", "")),
        timestamp=str(item.get("timestamp", "")),
        source=str(item.get("source", "")),
        topic=str(item.get("topic", "")),
        caption_entities_spacy=item.get("caption_entities_spacy", []),
        title=str(item.get("title", "")),
    )


def load_examples(split: str, limit: int | None = None) -> list[NewsClippingsExample]:
    annotations, metadata = load_split(split)
    if limit is not None:
        annotations = annotations[: max(0, limit)]
    return [build_example(annotation=a, metadata=metadata) for a in annotations]
