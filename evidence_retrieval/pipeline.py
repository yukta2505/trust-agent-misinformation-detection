"""End-to-end evidence retrieval pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .captioning import ImageCaptioner
from .config import AppConfig
from .entity_extraction import EntityExtractor
from .evidence_ranker import EvidenceRanker
from .historical_index import HistoricalEvidenceIndex
from .reverse_image_search import ReverseImageSearcher
from .web_search import WebEvidenceRetriever

LOGGER = logging.getLogger(__name__)


class EvidenceRetrievalPipeline:
    """Unified pipeline for multimodal evidence retrieval and ranking."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()
        self.captioner = ImageCaptioner(self.config)
        self.entity_extractor = EntityExtractor(self.config)
        self.reverse_searcher = ReverseImageSearcher(self.config)
        self.web_retriever = WebEvidenceRetriever(self.config)
        self.historical_index = HistoricalEvidenceIndex(self.config)
        self.ranker = EvidenceRanker(self.config)

    def run(
        self,
        image_path: str,
        claim: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute retrieval and ranking flow for a claim + image."""
        k = top_k or self.config.default_top_k
        LOGGER.info("Starting evidence retrieval pipeline for image=%s", image_path)

        # 1) Generate a caption for the image and extract entities from claim+caption.
        caption = self.captioner.generate_caption(image_path=image_path)
        combined_text = f"{claim} {caption}"
        entities = self.entity_extractor.extract_entities(combined_text)

        reverse_evidence: List[Dict[str, Any]] = []
        web_evidence: List[Dict[str, Any]] = []
        historical_evidence: List[Dict[str, Any]] = []

        try:
            # 2) Reverse-image search (SerpAPI-only in current configuration).
            reverse_evidence = self.reverse_searcher.search(image_path=image_path, top_k=k)
        except Exception as exc:
            LOGGER.warning("Reverse image retrieval failed: %s", exc)

        try:
            # 3) Web/news search from the claim, entities, and caption.
            web_evidence = self.web_retriever.search(
                claim=claim, entities=entities, caption=caption, top_k=k
            )
        except Exception as exc:
            LOGGER.warning("Web retrieval failed: %s", exc)

        try:
            # 4) Historical semantic search over indexed captions (if built).
            historical_evidence = self.historical_index.search(query=combined_text, top_k=k)
        except Exception as exc:
            LOGGER.warning("Historical retrieval failed: %s", exc)

        all_evidence: List[Dict[str, Any]] = reverse_evidence + web_evidence + historical_evidence
        # 5) Rank all evidence candidates together.
        ranked = self.ranker.rank(
            claim=claim,
            caption=caption,
            query_entities=entities,
            evidence_items=all_evidence,
            top_k=k,
        )

        return {
            "caption": caption,
            "entities": entities,
            "evidence": [
                {
                    "type": item.get("type"),
                    "title": item.get("title") or item.get("article_title") or item.get("caption"),
                    "url": item.get("url"),
                    "score": item.get("score"),
                    "source": item.get("source"),
                    "snippet": item.get("snippet"),
                }
                for item in ranked
            ],
        }
