"""Temporal Reasoning Agent — checks if the image timeline matches the claim."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from backend.config import Config
from backend.utils import clean_text
from backend.llm_client import get_llm_client

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Temporal Reasoning Agent for a misinformation detection system.

Your job: Determine whether the TIME and DATE in the claim is CONSISTENT with reality.

STEP 1 — Understand the claim type first:

TYPE A — "Old image being claimed as recent event"
  Example: "Beirut explosion today 2026" — but explosion was 2020
  → Check: does evidence show the event was actually in a different year?
  → If YES → temporal_score: 0.0 to 0.2

TYPE B — "Historical subject with a recent action"
  Example: "19th century painting sold for $17.9m" — painting IS old, sale IS recent
  Example: "Ancient temple restored in 2024" — temple IS old, restoration IS recent
  Example: "100-year-old document auctioned today" — document IS old, auction IS recent
  → These are CONSISTENT — the historical date and recent action are both true
  → temporal_score: 0.7 to 0.9

TYPE C — "Recent event with no date information"
  Example: "Protesters march in Paris" — no specific year mismatch found
  → temporal_score: 0.6 (benefit of doubt)

STEP 2 — Apply the right scoring:

CRITICAL RULE: If the claim mentions BOTH a historical period AND a recent action
(sale, auction, discovery, record, award, exhibition) — this is TYPE B.
Do NOT flag TYPE B as a temporal mismatch. Score 0.7+.

COMMON TIME-TRICK PATTERNS (TYPE A — flag these):
- Beirut explosion → actually happened August 2020, not 2025/2026
- Notre Dame fire → actually April 2019, not 2025/2026
- COVID lockdowns → 2020, not 2025/2026
- Kerala floods → 2018/2019, not 2025/2026
- Nepal earthquake → 2015, not 2025/2026
- Australia bushfires → 2019/2020, not 2025/2026
- George Floyd protests → 2020, not 2025/2026
- Yellow vest protests → 2018/2019, not 2025/2026

NOT time tricks (TYPE B — do not flag):
- Old painting/artwork sold/auctioned recently
- Historical document discovered or exhibited recently
- Ancient artifact sold at record price
- Old film/book winning recent award
- Historical monument recently renovated

You must respond ONLY with valid JSON — no markdown fences, no extra text.

Response schema:
{
  "claim_type": "TYPE_A or TYPE_B or TYPE_C",
  "claim_time_reference": "what time/date the claim mentions",
  "image_time_reference": "what time period the image is from",
  "time_gap_description": "explanation of temporal relationship",
  "is_temporally_consistent": true or false,
  "temporal_score": 0.0,
  "reasoning": "specific explanation"
}

temporal_score:
- 0.8–1.0 → consistent (TYPE B confirmed, or claim date matches evidence)
- 0.6–0.75 → neutral (TYPE C — no mismatch found)
- 0.3–0.5 → possible mismatch but uncertain
- 0.0–0.2 → clear TYPE A mismatch — evidence shows different year
"""


class TemporalReasoningAgent:
    def __init__(self, config: Config) -> None:
        # from openai import OpenAI
        # self.client = OpenAI(api_key=config.openai_api_key)
        if isinstance(config, tuple):
            config = config[0]
        self.config = config
        self.client = get_llm_client(config)

    def analyse(
        self,
        claim: str,
        caption: str,
        evidence_items: List[Dict[str, Any]],
        metadata_summary: str = "",
    ) -> Dict[str, Any]:

        evidence_lines = []
        for i, e in enumerate(evidence_items[:8]):
            ts = e.get("timestamp") or "unknown date"
            title = e.get("title", "")
            snippet = e.get("snippet", "")
            source = e.get("source", "")
            evidence_lines.append(
                f"[{i+1}] Date: {ts} | Source: {source}\n"
                f"    Title: {title}\n"
                f"    Snippet: {snippet}"
            )
        evidence_block = "\n".join(evidence_lines) or "(no evidence retrieved)"
        meta_block = f"\nIMAGE METADATA: {metadata_summary}" if metadata_summary else ""

        user_message = f"""Analyse temporal consistency for this fact-check:

CLAIM: {clean_text(claim)}

IMAGE CAPTION: {clean_text(caption)}{meta_block}

RETRIEVED EVIDENCE:
{evidence_block}

INSTRUCTIONS:
1. First identify the claim TYPE (A, B, or C) from the system prompt rules
2. For TYPE B (historical subject + recent action) → score HIGH (0.7-0.9)
3. For TYPE A (old image claimed as recent event) → score LOW (0.0-0.2)
4. For TYPE C (no clear date info) → score NEUTRAL (0.6)

Examples:
- "19th century painting sold for record price" → TYPE B → score 0.85
- "Beirut explosion today 2026" → TYPE A → score 0.0
- "Protesters march in city" → TYPE C → score 0.6

Return JSON only."""

        try:
            response = self.client.chat.completions.create(
                model=self.config.active_model,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            LOG.info(
                "Temporal Agent: type=%s score=%s | %s",
                result.get("claim_type", "?"),
                result.get("temporal_score"),
                result.get("time_gap_description", "")[:60],
            )
            return result

        except json.JSONDecodeError as exc:
            LOG.error("Temporal Agent JSON error: %s", exc)
            return {
                "claim_type": "TYPE_C",
                "claim_time_reference": "unknown",
                "image_time_reference": "unknown",
                "time_gap_description": "parse error",
                "is_temporally_consistent": True,
                "temporal_score": 0.6,
                "reasoning": f"Parse error: {exc}",
            }
        except Exception as exc:
            LOG.error("Temporal Agent API error: %s", exc)
            return {
                "claim_type": "TYPE_C",
                "claim_time_reference": "unknown",
                "image_time_reference": "unknown",
                "time_gap_description": "api error",
                "is_temporally_consistent": True,
                "temporal_score": 0.6,
                "reasoning": f"API error: {exc}",
            }