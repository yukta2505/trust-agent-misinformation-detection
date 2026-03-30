"""
Historical Evidence Index
==========================
Builds a FAISS vector index over historical image-caption records
(e.g. from the NewsCLIPpings dataset) so the pipeline can do
semantic similarity search at retrieval time.

This is an OPTIONAL component — the pipeline works without it.
It kicks in only if the index file exists on disk.

Usage
-----
# Build index once (offline step):

idx = HistoricalEvidenceIndex(Config())
idx.build("dataset/sample.jsonl")      # or .json / .csv

# Search (called automatically by the pipeline):
results = idx.search("protest outside parliament 2024", top_k=5)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import Config
from backend.utils import clean_text

LOG = logging.getLogger(__name__)


class FaissVectorIndex:
    """Thin wrapper around faiss.IndexFlatL2."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(dimension)

    def add(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected shape [N, {self.dimension}], got {vectors.shape}")
        self.index.add(vectors.astype(np.float32))

    def search(self, query: np.ndarray, top_k: int):
        if query.ndim != 2 or query.shape[1] != self.dimension:
            raise ValueError(f"Expected shape [N, {self.dimension}], got {query.shape}")
        return self.index.search(query.astype(np.float32), top_k)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal


class HistoricalEvidenceIndex:
    """
    Semantic search over historical caption records using FAISS + sentence-transformers.

    The index is built once from a dataset file and persisted to disk.
    On subsequent runs it loads from disk automatically.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.embedder = SentenceTransformer(self.config.sentence_model)
        self._faiss: Optional[FaissVectorIndex] = None
        self._metadata: List[Dict[str, Any]] = []

    # ── Data loading ─────────────────────────────────────────────────────────

    def _load_records(self, path: str) -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        suffix = p.suffix.lower()
        records: List[Dict[str, Any]] = []
        if suffix == ".jsonl":
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif suffix == ".json":
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON dataset must be a list of records.")
            records = data
        elif suffix == ".csv":
            with p.open(encoding="utf-8", newline="") as f:
                records.extend(csv.DictReader(f))
        else:
            raise ValueError(f"Unsupported format: {suffix} (use .jsonl/.json/.csv)")
        LOG.info("Loaded %d records from %s", len(records), path)
        return records

    # ── Build & persist ───────────────────────────────────────────────────────

    def build(self, dataset_path: str) -> None:
        """
        Build and save the FAISS index from a dataset file.

        Each record should have a 'caption' field (and optionally
        'image_path', 'article_title', 'source', 'timestamp').
        """
        records = self._load_records(dataset_path)
        if not records:
            raise ValueError("No records found — cannot build empty index.")

        captions = [clean_text(r.get("caption", "")) for r in records]
        LOG.info("Encoding %d captions...", len(captions))
        embeddings = self.embedder.encode(
            captions, convert_to_numpy=True, show_progress_bar=True
        ).astype(np.float32)

        self._faiss = FaissVectorIndex(dimension=embeddings.shape[1])
        self._faiss.add(embeddings)

        self._metadata = [
            {
                "image_path":    r.get("image_path"),
                "caption":       r.get("caption"),
                "article_title": r.get("article_title"),
                "source":        r.get("source"),
                "timestamp":     r.get("timestamp"),
                "type":          "historical",
            }
            for r in records
        ]

        self._save()
        LOG.info("Historical index built: %d vectors (dim=%d)", self._faiss.ntotal, embeddings.shape[1])

    def _save(self) -> None:
        if self._faiss is None:
            return
        idx_path = Path(self.config.historical_index_path)
        meta_path = Path(self.config.historical_meta_path)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss.index, str(idx_path))
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=True, indent=2)
        LOG.info("Saved index → %s | metadata → %s", idx_path, meta_path)

    # ── Load from disk ────────────────────────────────────────────────────────

    def _load(self) -> bool:
        idx_path = Path(self.config.historical_index_path)
        meta_path = Path(self.config.historical_meta_path)
        if not idx_path.exists() or not meta_path.exists():
            return False
        raw = faiss.read_index(str(idx_path))
        wrapper = FaissVectorIndex(dimension=raw.d)
        wrapper.index = raw
        self._faiss = wrapper
        with meta_path.open(encoding="utf-8") as f:
            self._metadata = json.load(f)
        LOG.info("Loaded historical index: %d vectors", self._faiss.ntotal)
        return True

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the index for records semantically similar to `query`.

        Automatically loads from disk if not yet loaded.
        Returns empty list (not an error) if the index doesn't exist yet.
        """
        if self._faiss is None:
            if not self._load():
                LOG.info("No historical index found — skipping.")
                return []

        q_emb = self.embedder.encode(
            [clean_text(query)], convert_to_numpy=True
        ).astype(np.float32)
        distances, indices = self._faiss.search(q_emb, top_k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            item = dict(self._metadata[idx])
            item["semantic_distance"] = float(dist)
            # Convert L2 distance to a 0-1 similarity score
            item["semantic_similarity"] = float(1.0 / (1.0 + dist))
            results.append(item)

        return results

    @property
    def is_built(self) -> bool:
        return Path(self.config.historical_index_path).exists()
