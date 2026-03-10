"""Shared helper functions for evidence retrieval components."""

from __future__ import annotations

import datetime as dt
import math
import re
from typing import Iterable, List, Optional

import numpy as np

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def clean_text(text: str) -> str:
    """Normalize text for search and similarity workflows."""
    text = text or ""
    text = re.sub(r"\s+", " ", text.strip())
    return text


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract lightweight keywords from text by filtering stopwords."""
    tokens = re.findall(r"[A-Za-z0-9]+", clean_text(text).lower())
    keywords: List[str] = []
    for token in tokens:
        if token in _STOPWORDS or len(token) < 3:
            continue
        if token not in keywords:
            keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def normalize_timestamp(timestamp: str | None) -> Optional[dt.datetime]:
    """Parse common timestamp formats to timezone-aware UTC datetime."""
    if not timestamp:
        return None
    ts = timestamp.strip()
    candidates = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in candidates:
        try:
            parsed = dt.datetime.strptime(ts, fmt)
            if fmt.endswith("Z"):
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            elif parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc)
        except ValueError:
            continue
    return None


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.asarray(list(vec_a), dtype=np.float32)
    b = np.asarray(list(vec_b), dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def recency_score(timestamp: str | None, half_life_days: int = 365) -> float:
    """Score recency between 0 and 1 with exponential decay."""
    parsed = normalize_timestamp(timestamp)
    if parsed is None:
        return 0.25
    now = dt.datetime.now(dt.timezone.utc)
    delta_days = max((now - parsed).days, 0)
    decay = math.exp(-math.log(2) * (delta_days / max(half_life_days, 1)))
    return float(min(max(decay, 0.0), 1.0))
