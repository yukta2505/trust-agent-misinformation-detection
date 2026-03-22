"""Temporal Reasoning Agent — checks if the image timeline matches the claim."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

from backend.config import Config
from backend.utils import clean_text

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Temporal Reasoning Agent for a misinformation detection system.

Your job: Determine whether the TIME and DATE information visible or implied in the image
(from its caption and evidence) is CONSISTENT with the claim being made.

A key misinformation pattern is: a real photo from one year/event is falsely claimed to
show a DIFFERENT year/event. Your job is to catch this.

You must respond ONLY with valid JSON — no markdown fences, no extra text.

Response schema:
{
  "claim_time_reference": "date/time/period mentioned in the claim",
  "image_time_reference": "date/time/period evident from caption + evidence",
  "time_gap_description": "e.g. '12 years apart', 'same period', 'unknown'",
  "is_temporally_consistent": true or false,
  "temporal_score": 0.0,
  "reasoning": "brief explanation of your temporal analysis"
}

temporal_score rules:
- 1.0  → no time conflict — image and claim are from the same period
- 0.7  → minor ambiguity, cannot confirm but no clear conflict
- 0.4  → evidence suggests a possible time mismatch
- 0.0  → clear time contradiction (e.g., image is from 2012, claim says 2024)

If no dates can be determined at all, return temporal_score: 0.5 (neutral).
"""


class TemporalReasoningAgent:
    """Uses GPT-4o to reason about temporal consistency between image and claim."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)

    def analyse(
        self,
        claim: str,
        caption: str,
        evidence_items: List[Dict[str, Any]],
        metadata_summary: str = "",
    ) -> Dict[str, Any]:
        evidence_lines = []
        for i, e in enumerate(evidence_items[:5]):
            ts = e.get("timestamp") or "unknown date"
            title = e.get("title", "")
            snippet = e.get("snippet", "")
            source = e.get("source", "")
            evidence_lines.append(
                f"[{i+1}] Date: {ts} | Source: {source}\n    {title} — {snippet}"
            )
        evidence_block = "\n".join(evidence_lines) or "(no evidence with timestamps available)"
        meta_block = f"\nIMAGE METADATA:\n{metadata_summary}" if metadata_summary else ""

        user_message = f"""Perform temporal analysis on the following:

CLAIM: {clean_text(claim)}

IMAGE CAPTION: {clean_text(caption)}{meta_block}

EVIDENCE WITH DATES:
{evidence_block}

Determine whether the time period implied in the image matches the claim, and return the JSON response."""

        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            LOG.info("Temporal Agent score: %s", result.get("temporal_score"))
            return result

        except json.JSONDecodeError as exc:
            LOG.error("Temporal Agent JSON parse error: %s", exc)
            return {
                "claim_time_reference": "unknown",
                "image_time_reference": "unknown",
                "time_gap_description": "could not determine",
                "is_temporally_consistent": None,
                "temporal_score": 0.5,
                "reasoning": "Parse error — defaulting to neutral score.",
            }
        except Exception as exc:
            LOG.error("Temporal Agent API error: %s", exc)
            return {
                "claim_time_reference": "unknown",
                "image_time_reference": "unknown",
                "time_gap_description": "could not determine",
                "is_temporally_consistent": None,
                "temporal_score": 0.5,
                "reasoning": f"API error: {exc}",
            }
