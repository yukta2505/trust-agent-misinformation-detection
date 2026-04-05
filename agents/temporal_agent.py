"""Temporal Reasoning Agent — checks if the image timeline matches the claim."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from backend.config import Config
from backend.utils import clean_text

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Temporal Reasoning Agent for a misinformation detection system.

Your job: Determine whether the TIME and DATE in the claim matches when the image was ACTUALLY taken.

CRITICAL INSTRUCTIONS:
1. Extract the year claimed (e.g. "2026", "today 2026", "March 2026")
2. Search the evidence for the ACTUAL year this event/image is from
3. If evidence shows the event happened in a DIFFERENT year → score 0.0 to 0.2
4. Be especially suspicious of claims saying recent years (2025/2026) for events that are famous from earlier years

COMMON TIME-TRICK PATTERNS to detect:
- Beirut explosion → actually happened August 2020, not 2025/2026
- Notre Dame fire → actually April 2019, not 2025/2026  
- COVID lockdowns → 2020, not 2025/2026
- Kerala floods → 2018/2019, not 2025/2026
- Nepal earthquake → 2015, not 2025/2026
- Australia bushfires → 2019/2020, not 2025/2026
- George Floyd protests → 2020, not 2025/2026

If the claim year contradicts the KNOWN year of the event → temporal_score MUST be 0.0 to 0.2

You must respond ONLY with valid JSON — no markdown fences, no extra text.

Response schema:
{
  "claim_time_reference": "exact date/year from the claim",
  "image_time_reference": "actual year this event occurred based on evidence or knowledge",
  "time_gap_description": "e.g. '6 years apart — event was 2020, claim says 2026'",
  "is_temporally_consistent": false,
  "temporal_score": 0.0,
  "reasoning": "specific explanation of time mismatch"
}

temporal_score rules:
- 0.9–1.0 → claim year matches actual event year confirmed by evidence
- 0.6–0.8 → no date information found, cannot confirm or deny
- 0.3–0.5 → some evidence suggests mismatch but uncertain
- 0.0–0.2 → clear mismatch — evidence or known facts show different year
"""


class TemporalReasoningAgent:
    # def __init__(self, config: Config) -> None:
    #     self.config = config
    #     from backend.llm_client import get_llm_client
    #     self.client = get_llm_client(config)

    def __init__(self, config: Config) -> None:
    # Guard against tuple being passed instead of Config object
        if isinstance(config, tuple):
            config = config[0]
        self.config = config
        from backend.llm_client import get_llm_client
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
1. What year does the CLAIM say this happened? Extract it precisely.
2. What year did this event ACTUALLY happen? Use evidence AND your knowledge.
3. Is there a year mismatch?

For famous events (Beirut explosion, Notre Dame fire, etc.), use your knowledge 
of when they actually occurred, even if evidence is sparse.

Return the JSON response."""

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
            LOG.info("Temporal Agent score: %s | %s",
                     result.get("temporal_score"),
                     result.get("time_gap_description", ""))
            return result

        except json.JSONDecodeError as exc:
            LOG.error("Temporal Agent JSON error: %s", exc)
            return {
                "claim_time_reference": "unknown",
                "image_time_reference": "unknown",
                "time_gap_description": "parse error",
                "is_temporally_consistent": None,
                "temporal_score": 0.5,
                "reasoning": f"Parse error: {exc}",
            }
        except Exception as exc:
            LOG.error("Temporal Agent API error: %s", exc)
            return {
                "claim_time_reference": "unknown",
                "image_time_reference": "unknown",
                "time_gap_description": "api error",
                "is_temporally_consistent": None,
                "temporal_score": 0.5,
                "reasoning": f"API error: {exc}",
            }