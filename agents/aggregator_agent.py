"""Aggregator Agent — synthesises all agent outputs into final verdict."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from backend.config import Config
from backend.llm_client import chat_with_fallback
from backend.utils import clean_text

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Aggregator Agent in TRUST-AGENT, a misinformation detection system.

You receive outputs from three specialist agents and a mathematical verdict.

VERDICT RULES:
- final_score >= 0.60 → "PRISTINE"  
- final_score < 0.60  → "OUT-OF-CONTEXT"
- You MUST use the mathematical verdict — do not override it

CONTEXT TRICK explanation guide:
If entity agent found mismatches in PEOPLE/PLACE/EVENT_TYPE, explain specifically:
- "The image shows X but the claim says Y"
- "The image is from [actual event] not [claimed event]"
- "The location shown is [actual] not [claimed]"

Be specific and factual. Do not use vague phrases like "temporal inconsistency" for 
context tricks — describe the ACTUAL mismatch found.

You must respond ONLY with valid JSON — no markdown fences.

{
  "verdict": "PRISTINE" or "OUT-OF-CONTEXT",
  "confidence_percent": 75,
  "explanation": "2-3 sentence plain English. Name the specific mismatch.",
  "key_evidence_for_verdict": ["specific finding 1", "specific finding 2", "specific finding 3"],
  "flags": ["specific red flag 1", "specific red flag 2"]
}
"""


class AggregatorAgent:
    def __init__(self, config: Config) -> None:
        self.config = config

    def aggregate(
        self,
        claim: str,
        caption: str,
        entity_result: Dict[str, Any],
        temporal_result: Dict[str, Any],
        credibility_result: Dict[str, Any],
        final_score: float,
        verdict: str,
    ) -> Dict[str, Any]:

        # Extract key findings for aggregator context
        contradictions = entity_result.get("contradictions", [])
        mismatch_dims = entity_result.get("mismatch_dimensions", [])
        context_mismatch = credibility_result.get("context_mismatch_detected", False)
        dominant_narrative = credibility_result.get("dominant_narrative", "")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Synthesise for final verdict:

CLAIM: {clean_text(claim)}
CAPTION: {clean_text(caption)}

MATHEMATICAL VERDICT: {verdict} (score={final_score:.3f}, threshold=0.60)

ENTITY AGENT (weight 35%): score={entity_result.get('entity_score', 0.5)}
  Mismatched dimensions: {mismatch_dims}
  Contradictions found: {contradictions}
  Reasoning: {entity_result.get('reasoning', '')}

TEMPORAL AGENT (weight 35%): score={temporal_result.get('temporal_score', 0.6)}
  Claim type: {temporal_result.get('claim_type', 'TYPE_C')}
  Gap: {temporal_result.get('time_gap_description', 'unknown')}
  Reasoning: {temporal_result.get('reasoning', '')}

CREDIBILITY AGENT (weight 30%): score={credibility_result.get('credibility_score', 0.55)}
  Context mismatch detected: {context_mismatch}
  Dominant narrative: {dominant_narrative}
  Reasoning: {credibility_result.get('reasoning', '')}

IMPORTANT: verdict MUST be "{verdict}".
If OUT-OF-CONTEXT, explain the SPECIFIC mismatch (wrong people/place/event type).
Return JSON only."""},
        ]

        try:
            raw = chat_with_fallback(self.config, messages, max_tokens=600)
            result = json.loads(raw)
            result["verdict"] = verdict  # enforce mathematical verdict always
            LOG.info(
                "Aggregator: %s (%s%%) | flags: %s",
                result.get("verdict"),
                result.get("confidence_percent"),
                result.get("flags", []),
            )
            return result
        except Exception as exc:
            LOG.error("Aggregator error: %s", exc)
            return {
                "verdict": verdict,
                "confidence_percent": int(abs(final_score - 0.5) * 200),
                "explanation": "Analysis completed with errors.",
                "key_evidence_for_verdict": contradictions[:3],
                "flags": contradictions[:2],
            }