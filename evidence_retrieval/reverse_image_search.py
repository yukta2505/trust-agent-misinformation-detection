"""Reverse image search with API-first strategy and CLIP fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .config import AppConfig
LOGGER = logging.getLogger(__name__)


class ReverseImageSearcher:
    """Search for visually similar evidence via SerpAPI reverse image."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig()

    def _upload_image(self, image_path: str) -> str:
        # Prefer ImgBB if configured; otherwise use anonymous 0x0.st upload.
        if self.config.imgbb_api_key:
            return self._upload_image_imgbb(image_path)
        return self._upload_image_0x0(image_path)

    def _upload_image_imgbb(self, image_path: str) -> str:
        endpoint = "https://api.imgbb.com/1/upload"
        with Path(image_path).open("rb") as f:
            resp = requests.post(
                endpoint,
                params={"key": self.config.imgbb_api_key, "expiration": 600},
                files={"image": f},
                timeout=self.config.request_timeout_sec,
            )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") or {}
        url = data.get("url")
        if not url:
            raise RuntimeError("ImgBB upload succeeded but no URL was returned.")
        return str(url)

    def _upload_image_0x0(self, image_path: str) -> str:
        endpoint = "https://0x0.st"
        with Path(image_path).open("rb") as f:
            resp = requests.post(
                endpoint,
                files={"file": f},
                timeout=self.config.request_timeout_sec,
            )
        resp.raise_for_status()
        url = resp.text.strip()
        if not url.startswith("http"):
            raise RuntimeError(f"0x0.st upload returned unexpected response: {url}")
        return url

    def _search_with_serpapi_reverse(self, image_path: str, top_k: int) -> List[Dict[str, Any]]:
        # SerpAPI requires a public image URL, so we upload the local image first.
        image_url = self._upload_image(image_path)
        params = {
            "engine": self.config.serpapi_reverse_engine,
            "image_url": image_url,
            "api_key": self.config.serpapi_key,
        }
        if self.config.serpapi_reverse_gl:
            params["gl"] = self.config.serpapi_reverse_gl
        if self.config.serpapi_reverse_hl:
            params["hl"] = self.config.serpapi_reverse_hl
        resp = requests.get(
            "https://serpapi.com/search.json",
            params=params,
            timeout=self.config.request_timeout_sec,
        )
        resp.raise_for_status()
        payload = resp.json()

        candidates: List[Dict[str, Any]] = []
        for key in ("image_results", "inline_images", "visual_matches"):
            items = payload.get(key)
            if isinstance(items, list) and items:
                candidates = items
                break

        results: List[Dict[str, Any]] = []
        for item in candidates[:top_k]:
            title = item.get("title") or item.get("snippet") or item.get("source")
            url = item.get("link") or item.get("page_link") or item.get("source")
            snippet = item.get("snippet") or item.get("source") or ""
            results.append(
                {
                    "type": "reverse_image",
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "SerpAPI Reverse Image",
                    "timestamp": None,
                }
            )
        return results

    def build_clip_index(self, dataset_path: str) -> None:
        """Deprecated: CLIP fallback removed in SerpAPI-only mode."""
        raise RuntimeError("CLIP fallback is disabled. Use SerpAPI reverse image search only.")

    def search(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search visually similar evidence via SerpAPI reverse image only."""
        if not self.config.serpapi_key:
            raise RuntimeError("SERPAPI_API_KEY is not configured.")
        return self._search_with_serpapi_reverse(image_path=image_path, top_k=top_k)
