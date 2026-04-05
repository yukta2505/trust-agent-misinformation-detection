"""
Evidence quality filter — removes irrelevant results before agents see them.
Place this file in your backend/ folder.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

LOG = logging.getLogger(__name__)

# ── Noise source patterns — always remove these ───────────────────────────────
_NOISE_TITLE_PATTERNS = [
    # Generic concept articles
    "metropolitan area", "what is a metro", "metropolitan area network",
    "definition", "merriam-webster", "britannica", "study.com",
    # Music/entertainment
    "official video", "official music video", "audio", "stream/download",
    "subscribe for more", "official audio", "lyric video", "music video",
    "ft.", "feat.", "album", "single", "spotify", "soundcloud",
    # Generic YouTube
    "youtube - topic", "vevo",
]

_NOISE_SOURCE_PATTERNS = [
    "genius.com", "spotify.com", "soundcloud.com", "bandcamp.com",
    "musixmatch.com", "azlyrics.com",
]


def _is_noise(item: Dict[str, Any]) -> bool:
    """Return True if this evidence item is clearly irrelevant."""
    title = (item.get("title") or "").lower()
    snippet = (item.get("snippet") or "").lower()
    source = (item.get("source") or "").lower()
    url = (item.get("url") or "").lower()

    # Check noise title patterns
    for pattern in _NOISE_TITLE_PATTERNS:
        if pattern in title or pattern in snippet:
            return True

    # Check noise sources
    for pattern in _NOISE_SOURCE_PATTERNS:
        if pattern in source or pattern in url:
            return True

    # YouTube music videos (URL contains watch but title has music keywords)
    if "youtube" in source and any(
        kw in title for kw in ["ft.", "feat.", "official", "audio", "album"]
    ):
        return True

    return False


def filter_relevant_evidence(
    evidence: List[Dict[str, Any]],
    claim: str,
    caption: str,
    min_overlap: int = 2,
) -> List[Dict[str, Any]]:
    """
    Keep only evidence items relevant to the claim.

    Two-stage filter:
    1. Remove known noise/garbage (music videos, dictionaries, etc.)
    2. Keep items with keyword overlap with claim
    """
    if not evidence:
        return []

    # ── Stage 1: Remove obvious noise ────────────────────────────────────────
    stage1 = []
    for e in evidence:
        if _is_noise(e):
            LOG.info("[filter] Noise removed: %s", e.get("title", "")[:60])
        else:
            stage1.append(e)

    LOG.info("[filter] After noise removal: %d / %d items", len(stage1), len(evidence))

    if not stage1:
        return []

    # ── Stage 2: Keyword overlap with claim ───────────────────────────────────
    stopwords = {
        "the", "and", "for", "are", "was", "were", "has", "have",
        "that", "this", "with", "from", "but", "not", "been", "its",
        "will", "they", "their", "can", "all", "hit", "today", "now",
        "amid", "after", "over", "into", "out", "more", "than", "when",
        "about", "been", "also", "had", "who", "one", "him", "her",
        "what", "just", "back", "would", "could", "should", "may",
    }

    # Clean claim — remove encoding garbage, lowercase
    claim_clean = re.sub(r'[^\x00-\x7F]+', '', claim).lower()
    claim_words = {
        w for w in re.findall(r'[a-z]{3,}', claim_clean)
        if w not in stopwords
    }

    LOG.info("[filter] Claim keywords for matching: %s", sorted(claim_words))

    if not claim_words:
        # Can't filter — return stage1 results as-is
        return stage1

    filtered = []
    for item in stage1:
        item_text = " ".join([
            str(item.get("title", "")),
            str(item.get("snippet", "")),
            str(item.get("source", "")),
        ]).lower()

        overlap = sum(1 for w in claim_words if w in item_text)
        item["relevance_overlap"] = overlap

        if overlap >= min_overlap:
            filtered.append(item)
            LOG.info("[filter] KEPT (overlap=%d): %s",
                     overlap, item.get("title", "")[:60])
        else:
            LOG.info("[filter] REMOVED (overlap=%d): %s",
                     overlap, item.get("title", "")[:60])

    # If filter removed everything, return stage1 to avoid empty evidence
    if not filtered:
        LOG.warning("[filter] All items filtered out — returning stage1 unfiltered")
        return stage1

    LOG.info("[filter] Final: %d relevant items kept", len(filtered))
    return filtered