"""
Evidence quality filter — removes irrelevant results before agents see them.

IMPORTANT: Be conservative with filtering.
It is better to pass slightly irrelevant evidence than to remove real evidence.
The agents (GPT/Groq) are smart enough to ignore irrelevant sources.
The filter should only remove OBVIOUS garbage like music videos and dictionary pages.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

LOG = logging.getLogger(__name__)

# ── Only remove OBVIOUS garbage ───────────────────────────────────────────────
# Keep this list SHORT — only things that are NEVER useful for fact-checking

_NOISE_TITLE_PATTERNS = [
    # Music content
    "official music video", "official video", "stream/download",
    "ft. kian", "duckwrth", "audio stream",
    # Generic concept definitions (not news)
    "what is a metropolitan area",
    "metropolitan area network definition",
    "merriam-webster definition",
    # Pure entertainment with no news value
    "lyrics genius", "azlyrics",
]

_NOISE_SOURCES = [
    "genius.com", "azlyrics.com", "spotify.com",
    "soundcloud.com", "bandcamp.com",
]


def _is_obvious_garbage(item: Dict[str, Any]) -> bool:
    """Only returns True for things that are NEVER useful for fact-checking."""
    title  = (item.get("title", "") or "").lower()
    source = (item.get("source", "") or "").lower()
    url    = (item.get("url", "") or "").lower()

    for pattern in _NOISE_TITLE_PATTERNS:
        if pattern in title:
            return True

    for noise in _NOISE_SOURCES:
        if noise in source or noise in url:
            return True

    # YouTube music videos specifically (not YouTube news)
    if "youtube" in source:
        music_signals = ["ft.", "feat.", "official video",
                         "official audio", "lyric video", "album"]
        if any(s in title for s in music_signals):
            return True

    return False


def filter_relevant_evidence(
    evidence: List[Dict[str, Any]],
    claim: str,
    caption: str,
    min_overlap: int = 1,
) -> List[Dict[str, Any]]:
    """
    Filter evidence — remove obvious garbage, keep everything else.

    Strategy:
    1. Remove obvious garbage (music videos, dictionary sites)
    2. Check keyword overlap with claim — keep if overlap >= 1
    3. If EVERYTHING gets filtered, return original unfiltered list
       (better to have noisy evidence than no evidence)
    """
    if not evidence:
        return []

    # ── Stage 1: Remove obvious garbage only ─────────────────────────────────
    stage1 = []
    for e in evidence:
        if _is_obvious_garbage(e):
            LOG.info("[filter] Removed garbage: %s", e.get("title", "")[:60])
        else:
            stage1.append(e)

    if not stage1:
        LOG.warning("[filter] All items were garbage — returning empty")
        return []

    # ── Stage 2: Keyword overlap check ────────────────────────────────────────
    # Clean claim — remove encoding garbage
    claim_clean = re.sub(r'[^\x00-\x7F]+', '', claim).lower()

    stopwords = {
        "the", "and", "for", "are", "was", "were", "has", "have",
        "that", "this", "with", "from", "but", "not", "been", "its",
        "will", "they", "their", "can", "all", "hit", "today", "now",
        "amid", "after", "over", "into", "out", "more", "than", "when",
        "about", "also", "had", "who", "one", "what", "just", "back",
        "shows", "show", "image", "images", "photo", "photograph",
        "picture", "video", "taken", "captured", "during",
    }

    claim_words = {
        w for w in re.findall(r'[a-z]{3,}', claim_clean)
        if w not in stopwords
    }

    LOG.debug("[filter] Claim keywords: %s", sorted(claim_words)[:15])

    # If no meaningful keywords extracted — return all stage1 results
    if not claim_words:
        LOG.info("[filter] No keywords extracted — returning %d items unfiltered",
                 len(stage1))
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
        else:
            LOG.debug("[filter] Low overlap (%d): %s",
                      overlap, item.get("title", "")[:60])

    # ── Safety net: never return empty if stage1 had results ─────────────────
    if not filtered and stage1:
        LOG.warning(
            "[filter] Keyword filter removed all %d items — "
            "returning unfiltered stage1 to preserve evidence",
            len(stage1)
        )
        return stage1

    LOG.info("[filter] Kept %d / %d evidence items (from %d total)",
             len(filtered), len(stage1), len(evidence))
    return filtered