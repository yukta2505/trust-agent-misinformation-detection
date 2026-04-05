"""Entity Analysis Agent — checks entity consistency between image and claim."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

from backend.config import Config
from backend.utils import clean_text
from backend.llm_client import get_llm_client

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Entity Analysis Agent for a misinformation detection system.

Your job: Given an image caption and a text claim, identify all named entities (people,
places, organisations, dates, events) in both, then check whether they are CONSISTENT
or CONTRADICTORY.

IMPORTANT RULES:
- If the caption does NOT contradict the claim → lean towards CONSISTENT (higher score)
- Only give a LOW score when there is a CLEAR, SPECIFIC contradiction
- Absence of confirming detail is NOT a contradiction
- A generic caption (e.g. "people holding signs") that does not conflict with the claim → score 0.6 or higher
- Only score below 0.4 if you can identify a specific entity mismatch (wrong country, wrong person, wrong organisation)

You must respond ONLY with valid JSON — no markdown fences, no extra text.

Response schema:
{
  "claim_entities": {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "EVENT": []},
  "caption_entities": {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "EVENT": []},
  "matches": ["entities found in both"],
  "contradictions": ["only SPECIFIC factual contradictions"],
  "entity_score": 0.0,
  "reasoning": "brief explanation"
}

entity_score:
- 0.9–1.0 → caption clearly confirms claim entities
- 0.6–0.8 → caption does not contradict claim (neutral/consistent)
- 0.3–0.5 → caption is vague and some entities are unverifiable
- 0.0–0.2 → caption SPECIFICALLY contradicts a key entity in the claim
"""


class EntityAnalysisAgent:
    def __init__(self, config: Config) -> None:
        self.config = config
        # self.client = OpenAI(api_key=config.openai_api_key)
        self.client = get_llm_client(config)

    def analyse(self, claim: str, caption: str,
                evidence_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        evidence_summary = "\n".join(
            f"- [{i+1}] {e.get('title','')}: {e.get('snippet','')}"
            for i, e in enumerate(evidence_items[:5])
        )
        user_message = f"""CLAIM: {clean_text(claim)}

IMAGE CAPTION: {clean_text(caption)}

EVIDENCE:
{evidence_summary or "(no external evidence retrieved)"}

Analyse entity consistency. Remember: no contradiction = consistent score (0.6+).
Return JSON only."""

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
            LOG.info("Entity Agent score: %s", result.get("entity_score"))
            return result
        except Exception as exc:
            LOG.error("Entity Agent error: %s", exc)
            return {"claim_entities": {}, "caption_entities": {},
                    "matches": [], "contradictions": [],
                    "entity_score": 0.5, "reasoning": f"Error: {exc}"}