"""Temporal Reasoning Agent — improved for miscaptioned detection."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from backend.config import Config
from backend.llm_client import chat_with_fallback
from backend.utils import clean_text

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Temporal Reasoning Agent for a misinformation detection system.

Your job: Determine if the TIME/DATE in the claim is consistent with reality.

════════════════════════════════════════════════════════
STEP 1 — CLASSIFY THE CLAIM TYPE:

TYPE A — Old event image falsely claimed as RECENT/CURRENT
  Key signal: Claim uses words like "today", "now", "currently", "2025", "2026"
  AND evidence shows the event actually happened years earlier
  Example: "Beirut explosion today 2026" (actually happened 2020)
  → temporal_score: 0.0 to 0.2

TYPE B — Claim correctly states a historical date or recent verified event
  Key signal: Claim has a specific past date AND that date is plausible/correct
  Example: "Paris Hilton selfie taken in 2006"
  Example: "NASA Mars Rover photo on May 7, 2022"
  Example: "Norwegian Navy missile test in 2013"
  Example: "China rocket launch December 2019"
  Example: "Jan 6 2020 Michigan protest"
  Example: "19th century painting sold for $17.9m"
  → temporal_score: 0.65 to 0.85 (the date is part of the true claim)

TYPE B_WRONG — Claim states a specific date BUT evidence shows a DIFFERENT date
  Key signal: Claim says event happened in year X, but evidence points to year Y
  Example: "Glastonbury 2015 garbage" (but photo is actually from 2016 Glastonbury)
  Example: "Jan 6 2020 Michigan protest" (actually a different location/date)
  → temporal_score: 0.20 to 0.40

TYPE C — No specific date or year in claim
  Example: "Image shows electric scooters abandoned by Mobike"
  Example: "Image shows athletes highlining"
  → temporal_score: 0.65 (neutral — nothing to check)

════════════════════════════════════════════════════════
CRITICAL RULES:

1. TYPE B historical dates: If a claim states a specific year (2006, 2013, 2015,
   2019, 2020, 2022 etc.) AND it's a factual description with no evidence 
   contradicting that date → score 0.65 to 0.80

2. TYPE A only applies when the claim says something is CURRENT/RECENT/TODAY
   but evidence shows it happened in a different year YEARS AGO.

3. TYPE B_WRONG: When evidence ACTIVELY shows the claimed date is wrong
   (e.g., claim says 2015 but all sources point to 2016) → score 0.20-0.40

4. NEVER score 0.0 for a claim with a historical date unless you have
   SPECIFIC evidence proving that date is wrong.

5. For MISCAPTIONED content (image is real but from different event):
   If evidence shows the image is ACTUALLY from event/year X but claim says event/year Y,
   AND they differ by more than 1 year OR are completely different events:
   → TYPE B_WRONG, score 0.20-0.35

KNOWN TIME TRICKS to detect (TYPE A):
- "Beirut explosion today/2025/2026" → actually Aug 2020
- "Notre Dame fire today/2025" → actually April 2019  
- "COVID lockdowns today 2025" → actually 2020
- "Kerala floods today 2025" → actually 2018/2019
- "Yellow vest protests today 2025" → actually 2018/2019

NOT time tricks (TYPE B — do not penalize):
- Any claim that states the correct historical year of an event
- Old paintings/artworks sold or auctioned recently
- Historical events described with their actual year

You must respond ONLY with valid JSON — no markdown fences.

Response schema:
{
  "claim_type": "TYPE_A|TYPE_B|TYPE_B_WRONG|TYPE_C",
  "claim_time_reference": "date/year from claim or 'none'",
  "image_time_reference": "actual time period or 'unknown'",
  "time_gap_description": "consistent / no date in claim / X years apart",
  "is_temporally_consistent": true,
  "temporal_score": 0.0,
  "reasoning": "specific explanation of classification and score"
}

temporal_score guide:
0.75–1.0 → TYPE B confirmed consistent, or strong evidence supports claim date
0.65     → TYPE C (no date) OR TYPE B (historical date, no contradiction found)
0.35–0.60 → Some doubt but uncertain
0.20–0.34 → TYPE B_WRONG: evidence shows different date/event than claimed
0.00–0.19 → TYPE A confirmed: evidence shows clearly different year/event
"""


class TemporalReasoningAgent:
    def __init__(self, config: Config) -> None:
        self.config = config

    def analyse(
        self,
        claim: str,
        caption: str,
        evidence_items: List[Dict[str, Any]],
        metadata_summary: str = "",
    ) -> Dict[str, Any]:

        evidence_lines = []
        for i, e in enumerate(evidence_items[:6]):
            ts = e.get("timestamp") or "unknown date"
            evidence_lines.append(
                f"[{i+1}] {ts} | {e.get('source','')} | "
                f"{e.get('title','')} — {e.get('snippet','')}"
            )
        evidence_block = "\n".join(evidence_lines) or "(no evidence retrieved)"
        meta_block = f"\nIMAGE METADATA: {metadata_summary}" if metadata_summary else ""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Analyse temporal consistency:

CLAIM: {clean_text(claim)}
IMAGE CAPTION: {clean_text(caption)}{meta_block}

EVIDENCE:
{evidence_block}

CLASSIFICATION GUIDE:
- Does the claim say something is happening TODAY/NOW/CURRENTLY? → Check for TYPE A
- Does the claim state a specific past year (e.g., "in 2006", "December 2019")? → TYPE B, score 0.65+
  UNLESS evidence shows a DIFFERENT specific date → TYPE B_WRONG, score 0.20-0.35
- No date at all? → TYPE C, score 0.65
- Only TYPE A gets very low scores (0.0-0.19), only if evidence confirms different year
- TYPE B_WRONG gets 0.20-0.35 when evidence actively contradicts claimed date/event

Return JSON only."""},
        ]

        try:
            raw = chat_with_fallback(self.config, messages, max_tokens=500)
            result = json.loads(raw)

            claim_type = result.get("claim_type", "TYPE_C")
            score = float(result.get("temporal_score", 0.65))

            # Safety overrides
            # TYPE C must be >= 0.60
            if claim_type == "TYPE_C" and score < 0.60:
                LOG.warning("TYPE_C score=%.2f → overriding to 0.65", score)
                result["temporal_score"] = 0.65
                result["reasoning"] += " [Override: TYPE_C must be >= 0.60]"

            # TYPE B must be >= 0.60 (not B_WRONG)
            if claim_type == "TYPE_B" and score < 0.60:
                LOG.warning("TYPE_B score=%.2f → overriding to 0.65", score)
                result["temporal_score"] = 0.65
                result["reasoning"] += " [Override: TYPE_B must be >= 0.60]"

            # TYPE B_WRONG floor: 0.15 (don't collapse to zero without strong evidence)
            if claim_type == "TYPE_B_WRONG" and score < 0.15:
                LOG.warning("TYPE_B_WRONG score=%.2f → flooring to 0.20", score)
                result["temporal_score"] = 0.20
                result["reasoning"] += " [Override: TYPE_B_WRONG floor 0.20]"

            # Low score without TYPE_A or TYPE_B_WRONG → override
            if score < 0.30 and claim_type not in ("TYPE_A", "TYPE_B_WRONG"):
                LOG.warning(
                    "Low score=%.2f but type=%s → overriding to 0.65", score, claim_type
                )
                result["temporal_score"] = 0.65
                result["claim_type"] = "TYPE_C"
                result["reasoning"] += " [Override: Low score without TYPE_A/B_WRONG evidence]"

            LOG.info(
                "Temporal Agent: type=%s score=%.3f | %s",
                result.get("claim_type"),
                result.get("temporal_score"),
                result.get("time_gap_description", "")[:60],
            )
            return result

        except Exception as exc:
            LOG.error("Temporal Agent error: %s", exc)
            return {
                "claim_type": "TYPE_C",
                "claim_time_reference": "none",
                "image_time_reference": "unknown",
                "time_gap_description": "error",
                "is_temporally_consistent": True,
                "temporal_score": 0.65,
                "reasoning": f"Error — neutral default: {exc}",
            }