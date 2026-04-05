"""Evidence retrieval: reverse image search + web/news search."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import parse, request

import requests

from backend.config import Config
from backend.utils import clean_text, extract_keywords

LOG = logging.getLogger(__name__)


class ReverseImageSearcher:
    """Upload image and run SerpAPI reverse-image search."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()

    # def _upload_imgbb(self, image_path: str) -> str:
    #     with Path(image_path).open("rb") as f:
    #         resp = requests.post(
    #             "https://api.imgbb.com/1/upload",
    #             params={"key": self.config.imgbb_api_key, "expiration": 600},
    #             files={"image": f},
    #             timeout=self.config.request_timeout,
    #         )
    #     resp.raise_for_status()
    #     url = (resp.json().get("data") or {}).get("url")
    #     if not url:
    #         raise RuntimeError("ImgBB: no URL in response")
    #     return url

    def _upload_imgbb(self, image_path: str) -> str:
        """Upload to ImgBB with retry on failure."""
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with Path(image_path).open("rb") as f:
                    resp = requests.post(
                        "https://api.imgbb.com/1/upload",
                        params={
                            "key": self.config.imgbb_api_key,
                            "expiration": 600,
                        },
                        files={"image": f},
                        timeout=self.config.request_timeout,
                    )
                resp.raise_for_status()
                url = (resp.json().get("data") or {}).get("url")
                if url:
                    LOG.info("ImgBB upload success on attempt %d", attempt + 1)
                    return url
                raise RuntimeError("ImgBB: no URL in response")
            except Exception as exc:
                LOG.warning("ImgBB attempt %d failed: %s", attempt + 1, exc)
                if attempt < max_retries - 1:
                    time.sleep(2)  # wait 2 seconds before retry
        raise RuntimeError(f"ImgBB upload failed after {max_retries} attempts")

    def _upload_0x0(self, image_path: str) -> str:
        with Path(image_path).open("rb") as f:
            resp = requests.post(
                "https://0x0.st",
                files={"file": f},
                timeout=self.config.request_timeout,
            )
        resp.raise_for_status()
        url = resp.text.strip()
        if not url.startswith("http"):
            raise RuntimeError(f"0x0.st unexpected: {url}")
        return url

    def _upload(self, image_path: str) -> str:
        if self.config.imgbb_api_key:
            return self._upload_imgbb(image_path)
        return self._upload_0x0(image_path)

    def search(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.config.serpapi_key:
            LOG.info("No SERPAPI key — skipping reverse image search.")
            return []
        try:
            image_url = self._upload(image_path)
            LOG.info("Image uploaded for reverse search: %s", image_url[:60])
        except Exception as exc:
            LOG.warning("Image upload failed: %s", exc)
            return []

        params = {
            "engine": "google_reverse_image",
            "image_url": image_url,
            "api_key": self.config.serpapi_key,
        }
        try:
            resp = requests.get(
                "https://serpapi.com/search.json",
                params=params,
                timeout=self.config.request_timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            LOG.warning("Reverse image SerpAPI call failed: %s", exc)
            return []

        items: List[Dict[str, Any]] = []
        for key in ("image_results", "inline_images", "visual_matches"):
            candidates = payload.get(key)
            if isinstance(candidates, list) and candidates:
                items = candidates
                break

        results = []
        for item in items[:top_k]:
            results.append({
                "type": "reverse_image",
                "title": item.get("title") or item.get("snippet") or "",
                "url": item.get("link") or item.get("page_link") or "",
                "snippet": item.get("snippet") or "",
                "source": "SerpAPI Reverse Image",
                "timestamp": None,
            })
        LOG.info("Reverse image search: %d results", len(results))
        return results


class WebSearcher:
    """Retrieve news/web evidence via SerpAPI or NewsAPI."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()

    def _build_query(
        self, claim: str, entities: Dict[str, List[str]], caption: str
    ) -> str:
        """
        Build a FOCUSED search query from the claim only.

        Key rule: use CLAIM text as primary source.
        Do NOT add caption words — they are often generic
        ('people holding signs', 'boat in water') and pollute results.
        Only add specific named entities from caption if they
        add geographic or event context.
        """
        claim_clean = clean_text(claim)

        # Remove encoding garbage characters
        import re
        claim_clean = re.sub(r'[^\x00-\x7F]+', '', claim_clean).strip()

        # Extract only high-value named entities (place names, org names)
        high_value = []
        for label in ("GPE", "ORG", "PERSON", "EVENT"):
            vals = entities.get(label, [])
            for v in vals[:2]:
                if len(v) > 2 and v not in claim_clean:
                    high_value.append(v)

        # Build query — claim is primary, entities only if they add info
        if high_value:
            query = f"{claim_clean} {' '.join(high_value[:2])}"
        else:
            query = claim_clean

        # Keep to 12 words max for best search results
        words = query.split()
        if len(words) > 12:
            query = " ".join(words[:12])

        LOG.info("Web search query: %s", query)
        return clean_text(query)

    def _serpapi(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.config.serpapi_key,
            "num": top_k,
        }
        url = f"https://serpapi.com/search.json?{parse.urlencode(params)}"
        with request.urlopen(
            request.Request(url), timeout=self.config.request_timeout
        ) as resp:
            payload = json.loads(resp.read())
        return [
            {
                "type": "web",
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "url": r.get("link", ""),
                "source": r.get("source") or "web",
                "timestamp": r.get("date"),
            }
            for r in payload.get("organic_results", [])[:top_k]
        ]

    def _newsapi(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": top_k,
            "apiKey": self.config.newsapi_key,
        }
        url = f"https://newsapi.org/v2/everything?{parse.urlencode(params)}"
        with request.urlopen(
            request.Request(url), timeout=self.config.request_timeout
        ) as resp:
            payload = json.loads(resp.read())
        return [
            {
                "type": "news",
                "title": a.get("title", ""),
                "snippet": a.get("description", ""),
                "url": a.get("url", ""),
                "source": (a.get("source") or {}).get("name", "news"),
                "timestamp": a.get("publishedAt"),
            }
            for a in payload.get("articles", [])[:top_k]
        ]

    def search(
        self,
        claim: str,
        entities: Dict[str, List[str]],
        caption: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        query = self._build_query(claim, entities, caption)

        if self.config.serpapi_key:
            try:
                results = self._serpapi(query, top_k)
                LOG.info("Web search: %d results", len(results))
                return results
            except Exception as exc:
                LOG.warning("SerpAPI web search failed: %s", exc)

        if self.config.newsapi_key:
            try:
                results = self._newsapi(query, top_k)
                LOG.info("NewsAPI: %d results", len(results))
                return results
            except Exception as exc:
                LOG.warning("NewsAPI failed: %s", exc)

        return []