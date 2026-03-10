"""FAISS helper abstraction for semantic search."""

from __future__ import annotations

from typing import Tuple

import faiss
import numpy as np


class FaissVectorIndex:
    """Thin wrapper around FAISS IndexFlatL2."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = self.create_index(dimension)

    @staticmethod
    def create_index(dimension: int) -> faiss.IndexFlatL2:
        """Create an L2 index with fixed vector dimension."""
        return faiss.IndexFlatL2(dimension)

    def add_vectors(self, vectors: np.ndarray) -> None:
        """Add vectors of shape [N, D] to the index."""
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected vectors with shape [N, {self.dimension}], got {tuple(vectors.shape)}"
            )
        self.index.add(vectors.astype(np.float32))

    def search_vectors(self, query_vectors: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search top-k nearest neighbors for query vectors."""
        if query_vectors.ndim != 2 or query_vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query shape [N, {self.dimension}], got {tuple(query_vectors.shape)}"
            )
        distances, indices = self.index.search(query_vectors.astype(np.float32), top_k)
        return distances, indices
