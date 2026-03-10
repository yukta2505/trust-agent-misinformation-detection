"""Historical evidence indexing and retrieval with sentence embeddings + FAISS."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import AppConfig
from .faiss_index import FaissVectorIndex
from .utils import clean_text

LOGGER = logging.getLogger(__name__)


class HistoricalEvidenceIndex:
    """Index NewsCLIPpings-like caption records for semantic search."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.embedder = SentenceTransformer(self.config.sentence_model_name)
        self.index: Optional[FaissVectorIndex] = None
        self.metadata: List[Dict[str, Any]] = []

    def _load_records(self, dataset_path: str) -> List[Dict[str, Any]]:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        suffix = path.suffix.lower()
        records: List[Dict[str, Any]] = []
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                records = payload
            else:
                raise ValueError("JSON dataset must contain a list of records.")
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                records.extend(reader)
        else:
            raise ValueError("Unsupported dataset format. Use .jsonl, .json, or .csv")
        return records

    def build_index(self, dataset_path: str) -> None:
        """Build FAISS index from historical caption records."""
        records = self._load_records(dataset_path)
        if not records:
            raise ValueError("No records found in historical dataset.")

        captions = [clean_text(r.get("caption", "")) for r in records]
        embeddings = self.embedder.encode(captions, convert_to_numpy=True).astype(np.float32)
        self.index = FaissVectorIndex(dimension=embeddings.shape[1])
        self.index.add_vectors(embeddings)
        self.metadata = [
            {
                "image_path": r.get("image_path"),
                "caption": r.get("caption"),
                "article_title": r.get("article_title"),
                "source": r.get("source"),
                "timestamp": r.get("timestamp"),
                "type": "historical",
            }
            for r in records
        ]
        self._persist()
        LOGGER.info("Historical index built with %d records", len(self.metadata))

    def _persist(self) -> None:
        if self.index is None:
            return
        Path(self.config.historical_index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index.index, self.config.historical_index_path)
        with Path(self.config.historical_metadata_path).open("w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=True, indent=2)

    def _load_if_available(self) -> bool:
        idx_path = Path(self.config.historical_index_path)
        meta_path = Path(self.config.historical_metadata_path)
        if not idx_path.exists() or not meta_path.exists():
            return False
        raw_index = faiss.read_index(str(idx_path))
        wrapper = FaissVectorIndex(dimension=raw_index.d)
        wrapper.index = raw_index
        self.index = wrapper
        with meta_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        return True

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search historical records by semantic caption similarity."""
        if self.index is None and not self._load_if_available():
            raise RuntimeError("Historical index is not built. Call build_index(dataset_path) first.")

        assert self.index is not None
        query_emb = self.embedder.encode([clean_text(query)], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search_vectors(query_emb, top_k=top_k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[idx])
            item["semantic_distance"] = float(dist)
            item["semantic_similarity"] = float(1.0 / (1.0 + dist))
            results.append(item)
        return results
