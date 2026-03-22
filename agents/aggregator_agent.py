"""Aggregator Agent — fuses agent scores and generates the final verdict."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from openai import OpenAI

from backend.config import Config
from backend.utils import clean_text

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Aggregator Agent in TRUST-AGENT, an out-of-context
misinformation detection system.

You receive the outputs of three specialist agents:
  1. Entity Analysis Agent  — found entity matches/contradictions
  2. Temporal Reasoning Agent — assessed time consistency
  3. Source Credibility Agent — evaluated trustworthiness of sources

Your job: Synthesise their findings into a final, human-readable verdict.

You must respond ONLY with valid JSON — no markdown fences, no extra text.

Response schema:
{
  "verdict": "PRISTINE" or "OUT-OF-CONTEXT",
  "confidence_percent": 85,
  "explanation": "2-4 sentence plain-English explanation a non-expert can understand",
  "key_evidence_for_verdict": ["bullet 1", "bullet 2", "bullet 3"],
  "flags": ["list of specific red flags found, or empty list if pristine"]
}

Verdict rules:
- "PRISTINE"       → the image appears to be correctly used in context
- "OUT-OF-CONTEXT" → the image is a real photo used with a misleading claim

confidence_percent: integer 0-100 representing how certain the system is.
Be direct, factual, and avoid jargon. The explanation must help the user understand
WHY the verdict was reached.
"""


class AggregatorAgent:
    """Uses GPT-4o to synthesise all agent outputs into a final verdict."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)

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
        user_message = f"""Synthesise the following analysis for a final verdict.

ORIGINAL CLAIM: {clean_text(claim)}
IMAGE CAPTION: {clean_text(caption)}

MATHEMATICAL VERDICT: {verdict} (weighted score = {final_score:.3f})

ENTITY AGENT FINDINGS:
  Score: {entity_result.get('entity_score', 0.5)}
  Matches: {entity_result.get('matches', [])}
  Contradictions: {entity_result.get('contradictions', [])}
  Reasoning: {entity_result.get('reasoning', '')}

TEMPORAL AGENT FINDINGS:
  Score: {temporal_result.get('temporal_score', 0.5)}
  Claim time: {temporal_result.get('claim_time_reference', 'unknown')}
  Image time: {temporal_result.get('image_time_reference', 'unknown')}
  Gap: {temporal_result.get('time_gap_description', 'unknown')}
  Consistent: {temporal_result.get('is_temporally_consistent')}
  Reasoning: {temporal_result.get('reasoning', '')}

CREDIBILITY AGENT FINDINGS:
  Score: {credibility_result.get('credibility_score', 0.5)}
  Cross-source agreement: {credibility_result.get('cross_source_agreement', 'INSUFFICIENT')}
  Dominant narrative: {credibility_result.get('dominant_narrative', '')}
  Reasoning: {credibility_result.get('reasoning', '')}

Produce the final JSON verdict."""

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
            LOG.info(
                "Aggregator verdict: %s (confidence: %s%%)",
                result.get("verdict"),
                result.get("confidence_percent"),
            )
            return result

        except json.JSONDecodeError as exc:
            LOG.error("Aggregator JSON parse error: %s", exc)
            return {
                "verdict": verdict,
                "confidence_percent": int(abs(final_score - 0.5) * 200),
                "explanation": "Automated analysis completed. Manual review recommended.",
                "key_evidence_for_verdict": [],
                "flags": [],
            }
        except Exception as exc:
            LOG.error("Aggregator API error: %s", exc)
            return {
                "verdict": verdict,
                "confidence_percent": int(abs(final_score - 0.5) * 200),
                "explanation": f"Analysis completed with partial results. Error: {exc}",
                "key_evidence_for_verdict": [],
                "flags": [],
            }
