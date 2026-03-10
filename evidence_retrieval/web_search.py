"""Web/news evidence retrieval via SerpAPI or NewsAPI."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from urllib import parse, request

from .config import AppConfig
from .utils import clean_text, extract_keywords

LOGGER = logging.getLogger(__name__)


class WebEvidenceRetriever:
    """Retrieve relevant web/news evidence from search APIs."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()

    def _build_query(self, claim: str, entities: Dict[str, List[str]], caption: str) -> str:
        entity_terms: List[str] = []
        for values in entities.values():
            entity_terms.extend(values[:3])
        query_parts = [claim] + entity_terms + extract_keywords(caption, max_keywords=5)
        query = clean_text(" ".join(str(p) for p in query_parts if p))
        return query

    def _search_serpapi(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.config.serpapi_key,
            "num": top_k,
        }
        url = f"https://serpapi.com/search.json?{parse.urlencode(params)}"
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=self.config.request_timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        results: List[Dict[str, Any]] = []
        for item in payload.get("organic_results", [])[:top_k]:
            results.append(
                {
                    "type": "news",
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "url": item.get("link"),
                    "source": item.get("source") or "web",
                    "timestamp": item.get("date"),
                }
            )
        return results

    def _search_newsapi(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": top_k,
            "apiKey": self.config.newsapi_key,
        }
        url = f"https://newsapi.org/v2/everything?{parse.urlencode(params)}"
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=self.config.request_timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        articles = payload.get("articles", [])
        return [
            {
                "type": "news",
                "title": article.get("title"),
                "snippet": article.get("description"),
                "url": article.get("url"),
                "source": (article.get("source") or {}).get("name", "news"),
                "timestamp": article.get("publishedAt"),
            }
            for article in articles[:top_k]
        ]

    def search(
        self, claim: str, entities: Dict[str, List[str]], caption: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search web/news evidence with API fallback order."""
        query = self._build_query(claim=claim, entities=entities, caption=caption)
        LOGGER.info("Web evidence query: %s", query)
        if self.config.serpapi_key:
            try:
                return self._search_serpapi(query=query, top_k=top_k)
            except Exception as exc:
                LOGGER.warning("SerpAPI failed, falling back to NewsAPI: %s", exc)
        if self.config.newsapi_key:
            try:
                return self._search_newsapi(query=query, top_k=top_k)
            except Exception as exc:
                LOGGER.warning("NewsAPI failed: %s", exc)
        LOGGER.info("No web/news API key configured; returning empty web evidence.")
        return []
