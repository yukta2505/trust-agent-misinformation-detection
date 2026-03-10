"""Evidence ranking logic for multimodal retrieval results."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from sentence_transformers import SentenceTransformer

from .config import AppConfig
from .utils import clean_text, recency_score

LOGGER = logging.getLogger(__name__)


class EvidenceRanker:
    """Rank evidence by semantic similarity, entity overlap, and recency."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.embedder = SentenceTransformer(self.config.sentence_model_name)

    @staticmethod
    def _flatten_entities(entities: Dict[str, List[str]]) -> Set[str]:
        flat: Set[str] = set()
        for values in entities.values():
            for value in values:
                v = value.strip().lower()
                if v:
                    flat.add(v)
        return flat

    @staticmethod
    def _evidence_text(item: Dict[str, Any]) -> str:
        return clean_text(
            " ".join(
                str(item.get(key, ""))
                for key in ("title", "snippet", "caption", "article_title", "source")
            )
        )

    def rank(
        self,
        claim: str,
        caption: str,
        query_entities: Dict[str, List[str]],
        evidence_items: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Rank evidence items and return top-k with score breakdown."""
        if not evidence_items:
            return []

        query_text = clean_text(f"{claim} {caption}")
        query_emb = self.embedder.encode([query_text], convert_to_numpy=True)[0]
        query_entity_set = self._flatten_entities(query_entities)

        scored: List[Dict[str, Any]] = []
        for item in evidence_items:
            text = self._evidence_text(item)
            if not text:
                continue
            item_emb = self.embedder.encode([text], convert_to_numpy=True)[0]
            semantic = float(
                (query_emb @ item_emb)
                / ((query_emb @ query_emb) ** 0.5 * (item_emb @ item_emb) ** 0.5 + 1e-8)
            )
            semantic = max(min((semantic + 1.0) / 2.0, 1.0), 0.0)

            item_lc = text.lower()
            overlap_count = sum(1 for ent in query_entity_set if ent in item_lc)
            entity_overlap = overlap_count / max(len(query_entity_set), 1)
            recency = recency_score(item.get("timestamp"))
            score = 0.5 * semantic + 0.3 * entity_overlap + 0.2 * recency

            ranked = dict(item)
            ranked["semantic_similarity"] = round(semantic, 4)
            ranked["entity_overlap"] = round(entity_overlap, 4)
            ranked["recency_score"] = round(recency, 4)
            ranked["score"] = round(float(score), 4)
            scored.append(ranked)

        scored.sort(key=lambda x: x["score"], reverse=True)
        LOGGER.info("Ranked %d evidence items", len(scored))
        return scored[:top_k]
