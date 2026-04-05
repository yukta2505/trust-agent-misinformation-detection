"""Source Credibility Agent — evaluates the trustworthiness of evidence sources."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

from backend.config import Config
from backend.utils import clean_text
from backend.llm_client import get_llm_client

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Source Credibility Agent for a misinformation detection system.

Your job: Evaluate whether the evidence sources support or contradict the claim.

IMPORTANT RULES:
- If NO evidence was retrieved → return credibility_score: 0.55 (neutral, not suspicious)
- Absence of evidence is NOT proof of misinformation
- If sources are irrelevant (not related to the claim topic) → treat as no evidence (score 0.55)
- Only give LOW score when credible sources SPECIFICALLY contradict the claim
- If even ONE credible source (BBC, Reuters, AP, AFP, government) supports the claim → score 0.75+

You must respond ONLY with valid JSON — no markdown fences, no extra text.

Response schema:
{
  "sources_evaluated": [
    {"source": "name", "credibility_tier": "HIGH|MEDIUM|LOW|UNKNOWN",
     "supports_claim": true, "reason": "one line"}
  ],
  "cross_source_agreement": "AGREE|DISAGREE|MIXED|INSUFFICIENT",
  "dominant_narrative": "what credible sources say",
  "credibility_score": 0.0,
  "reasoning": "overall assessment"
}

credibility_score:
- 0.8–1.0 → credible sources confirm the claim
- 0.6–0.75 → no sources found OR sources are irrelevant (neutral)
- 0.35–0.55 → mixed signals, some doubt
- 0.0–0.3 → credible sources directly contradict the claim

HIGH: BBC, Reuters, AP, AFP, NYT, Washington Post, .gov, Al Jazeera, major broadcasters
MEDIUM: regional news, Wikipedia, established outlets
LOW: social media, unknown blogs, tabloids
"""


class SourceCredibilityAgent:
    def __init__(self, config: Config) -> None:
        self.config = config
        # self.client = OpenAI(api_key=config.openai_api_key)
        self.client = get_llm_client(config)

    def analyse(self, claim: str, caption: str,
                evidence_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        source_lines = []
        for i, e in enumerate(evidence_items[:6]):
            source = e.get("source") or e.get("url") or "unknown"
            source_lines.append(
                f"[{i+1}] Source: {source}\n    {e.get('title','')}: {e.get('snippet','')}"
            )
        source_block = "\n".join(source_lines) or "(no sources retrieved)"

        user_message = f"""CLAIM: {clean_text(claim)}
IMAGE CAPTION: {clean_text(caption)}

RETRIEVED SOURCES:
{source_block}

Assess credibility. No evidence = neutral (0.55), not suspicious.
Return JSON only."""

        try:
            response = self.client.chat.completions.create(
                #model=self.config.openai_model,
                model=self.config.active_model,
                max_tokens=700,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content.strip())
            LOG.info("Credibility Agent score: %s", result.get("credibility_score"))
            return result
        except Exception as exc:
            LOG.error("Credibility Agent error: %s", exc)
            return {"sources_evaluated": [], "cross_source_agreement": "INSUFFICIENT",
                    "dominant_narrative": "No sources available",
                    "credibility_score": 0.55, "reasoning": f"Error: {exc}"}