"""Aggregator Agent — fuses agent scores and generates the final verdict."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from openai import OpenAI

from backend.config import Config
from backend.utils import clean_text
from backend.llm_client import get_llm_client

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Aggregator Agent in TRUST-AGENT, an out-of-context misinformation detection system.

You receive outputs from three specialist agents and a weighted score.

CRITICAL RULES:
- The mathematical verdict is based on a weighted score threshold of 0.60
- If final_score >= 0.60 → PRISTINE (the claim appears valid)
- If final_score < 0.60 → OUT-OF-CONTEXT (the claim is likely misleading)
- You MUST respect the mathematical verdict — do not override it
- When evidence is limited, low confidence scores (50-65%) are appropriate
- Only flag as OUT-OF-CONTEXT with high confidence when there is SPECIFIC contradicting evidence

You must respond ONLY with valid JSON — no markdown fences, no extra text.

Response schema:
{
  "verdict": "PRISTINE" or "OUT-OF-CONTEXT",
  "confidence_percent": 75,
  "explanation": "2-3 sentence plain English explanation. Be specific about what was found.",
  "key_evidence_for_verdict": ["specific finding 1", "specific finding 2"],
  "flags": ["only real specific red flags, empty list if PRISTINE or uncertain"]
}

confidence_percent:
- 85-100: clear strong evidence either way
- 65-84: moderate evidence
- 50-64: limited evidence, uncertain
- Below 50: very uncertain
"""


class AggregatorAgent:
    def __init__(self, config: Config) -> None:
        self.config = config
        # self.client = OpenAI(api_key=config.openai_api_key)
        self.client = get_llm_client(config)

    def aggregate(self, claim: str, caption: str,
                  entity_result: Dict[str, Any], temporal_result: Dict[str, Any],
                  credibility_result: Dict[str, Any],
                  final_score: float, verdict: str) -> Dict[str, Any]:

        user_message = f"""ORIGINAL CLAIM: {clean_text(claim)}
IMAGE CAPTION: {clean_text(caption)}

MATHEMATICAL VERDICT: {verdict} (weighted score = {final_score:.3f}, threshold = 0.60)

ENTITY AGENT (weight 35%):
  Score: {entity_result.get('entity_score', 0.5)}
  Matches: {entity_result.get('matches', [])}
  Contradictions: {entity_result.get('contradictions', [])}
  Reasoning: {entity_result.get('reasoning', '')}

TEMPORAL AGENT (weight 35%):
  Score: {temporal_result.get('temporal_score', 0.6)}
  Claim time: {temporal_result.get('claim_time_reference', 'unknown')}
  Image time: {temporal_result.get('image_time_reference', 'unknown')}
  Consistent: {temporal_result.get('is_temporally_consistent')}
  Reasoning: {temporal_result.get('reasoning', '')}

CREDIBILITY AGENT (weight 30%):
  Score: {credibility_result.get('credibility_score', 0.55)}
  Agreement: {credibility_result.get('cross_source_agreement', 'INSUFFICIENT')}
  Narrative: {credibility_result.get('dominant_narrative', '')}
  Reasoning: {credibility_result.get('reasoning', '')}

Generate the final verdict JSON. You MUST use verdict="{verdict}" as determined by the score."""

        try:
            response = self.client.chat.completions.create(
                # model=self.config.openai_model,
                model=self.config.active_model,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content.strip())
            # Always enforce mathematical verdict
            result["verdict"] = verdict
            LOG.info("Aggregator: %s (%s%%)", result.get("verdict"),
                     result.get("confidence_percent"))
            return result
        except Exception as exc:
            LOG.error("Aggregator error: %s", exc)
            return {"verdict": verdict,
                    "confidence_percent": int(abs(final_score - 0.5) * 200),
                    "explanation": "Analysis completed.",
                    "key_evidence_for_verdict": [], "flags": []}