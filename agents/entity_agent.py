"""Entity Analysis Agent — checks entity consistency between image and claim."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

from backend.config import Config
from backend.utils import clean_text

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Entity Analysis Agent for a misinformation detection system.

Your job: Given an image caption and a text claim, identify all named entities (people,
places, organisations, dates, events) in both, then check whether they are CONSISTENT
or CONTRADICTORY.

You must respond ONLY with valid JSON — no markdown fences, no extra text.

Response schema:
{
  "claim_entities": {
    "PERSON": ["..."],
    "ORG": ["..."],
    "GPE": ["..."],
    "DATE": ["..."],
    "EVENT": ["..."]
  },
  "caption_entities": {
    "PERSON": ["..."],
    "ORG": ["..."],
    "GPE": ["..."],
    "DATE": ["..."],
    "EVENT": ["..."]
  },
  "matches": ["entity that appears in both"],
  "contradictions": ["entity that contradicts or conflicts"],
  "entity_score": 0.0,
  "reasoning": "brief explanation"
}

entity_score rules:
- 1.0  → all key entities match perfectly
- 0.5  → partial match / some missing entities
- 0.0  → clear contradictions (wrong person, wrong place, wrong date)
"""


class EntityAnalysisAgent:
    """Uses GPT-4o to verify entity consistency between image caption and claim."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)

    def analyse(
        self,
        claim: str,
        caption: str,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        evidence_summary = "\n".join(
            f"- [{i+1}] {e.get('title', '')} | {e.get('snippet', '')}"
            for i, e in enumerate(evidence_items[:3])
        )

        user_message = f"""Analyse the following:

CLAIM: {clean_text(claim)}

IMAGE CAPTION: {clean_text(caption)}

TOP EVIDENCE:
{evidence_summary or "(no external evidence available)"}

Check entity consistency and return the JSON response."""

        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            LOG.info("Entity Agent score: %s", result.get("entity_score"))
            return result

        except json.JSONDecodeError as exc:
            LOG.error("Entity Agent JSON parse error: %s", exc)
            return {
                "claim_entities": {}, "caption_entities": {},
                "matches": [], "contradictions": [],
                "entity_score": 0.5,
                "reasoning": "Parse error — defaulting to neutral score.",
            }
        except Exception as exc:
            LOG.error("Entity Agent API error: %s", exc)
            return {
                "claim_entities": {}, "caption_entities": {},
                "matches": [], "contradictions": [],
                "entity_score": 0.5,
                "reasoning": f"API error: {exc}",
            }
