"""Shared utility functions."""

from __future__ import annotations

import datetime as dt
import math
import re
from typing import Iterable, List, Optional

import numpy as np

_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with",
}


def clean_text(text: str) -> str:
    text = text or ""
    return re.sub(r"\s+", " ", text.strip())


def extract_keywords(text: str, max_kw: int = 10) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", clean_text(text).lower())
    kws: List[str] = []
    for t in tokens:
        if t in _STOP or len(t) < 3 or t in kws:
            continue
        kws.append(t)
        if len(kws) >= max_kw:
            break
    return kws


def parse_timestamp(ts: Optional[str]) -> Optional[dt.datetime]:
    if not ts:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = dt.datetime.strptime(ts.strip(), fmt)
            if fmt.endswith("Z"):
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            elif parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc)
        except ValueError:
            continue
    return None


def recency_score(timestamp: Optional[str], half_life_days: int = 365) -> float:
    parsed = parse_timestamp(timestamp)
    if parsed is None:
        return 0.25
    delta = max((dt.datetime.now(dt.timezone.utc) - parsed).days, 0)
    return float(min(max(math.exp(-math.log(2) * (delta / max(half_life_days, 1))), 0.0), 1.0))


def cosine_sim(a: Iterable[float], b: Iterable[float]) -> float:
    va = np.asarray(list(a), dtype=np.float32)
    vb = np.asarray(list(b), dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0
