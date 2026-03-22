"""Source Credibility Agent — evaluates the trustworthiness of evidence sources."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from openai import OpenAI

from backend.config import Config
from backend.utils import clean_text

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Source Credibility Agent for a misinformation detection system.

Your job: Evaluate the credibility of the sources in the evidence retrieved for a claim.
You assess whether the evidence comes from trusted outlets, cross-source agreement,
and whether the sources support or contradict the claim.

You must respond ONLY with valid JSON — no markdown fences, no extra text.

Response schema:
{
  "sources_evaluated": [
    {
      "source": "source name or domain",
      "credibility_tier": "HIGH | MEDIUM | LOW | UNKNOWN",
      "supports_claim": true or false or null,
      "reason": "one-line rationale"
    }
  ],
  "cross_source_agreement": "AGREE | DISAGREE | MIXED | INSUFFICIENT",
  "dominant_narrative": "what the majority of credible sources say",
  "credibility_score": 0.0,
  "reasoning": "overall credibility assessment summary"
}

credibility_score rules:
- 1.0  → multiple HIGH-credibility sources all AGREE and support the claim
- 0.7  → mostly credible sources, minor disagreement
- 0.5  → mixed credibility, insufficient sources, or no external sources
- 0.2  → credible sources CONTRADICT the claim
- 0.0  → all credible sources directly debunk the claim

HIGH credibility: Reuters, AP, BBC, NYT, Washington Post, government sites (.gov),
                  peer-reviewed journals, major broadcast outlets
MEDIUM: regional news outlets, established blogs, Wikipedia
LOW: social media posts, unknown domains, tabloids
UNKNOWN: no identifiable source
"""


class SourceCredibilityAgent:
    """Uses GPT-4o to assess trustworthiness of evidence sources."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)

    def analyse(
        self,
        claim: str,
        caption: str,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        source_lines = []
        for i, e in enumerate(evidence_items[:6]):
            source = e.get("source") or e.get("url") or "unknown"
            title = e.get("title", "")
            snippet = e.get("snippet", "")
            source_lines.append(
                f"[{i+1}] Source: {source}\n    Title: {title}\n    Snippet: {snippet}"
            )
        source_block = "\n".join(source_lines) or "(no external sources retrieved)"

        user_message = f"""Evaluate source credibility for the following claim:

CLAIM: {clean_text(claim)}

IMAGE CAPTION: {clean_text(caption)}

RETRIEVED EVIDENCE SOURCES:
{source_block}

Assess each source's credibility tier, whether it supports/contradicts the claim,
and compute an overall credibility score. Return the JSON response."""

        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                max_tokens=900,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content.strip()
            result = json.loads(raw)
            LOG.info("Credibility Agent score: %s", result.get("credibility_score"))
            return result

        except json.JSONDecodeError as exc:
            LOG.error("Credibility Agent JSON parse error: %s", exc)
            return {
                "sources_evaluated": [],
                "cross_source_agreement": "INSUFFICIENT",
                "dominant_narrative": "Could not determine",
                "credibility_score": 0.5,
                "reasoning": "Parse error — defaulting to neutral score.",
            }
        except Exception as exc:
            LOG.error("Credibility Agent API error: %s", exc)
            return {
                "sources_evaluated": [],
                "cross_source_agreement": "INSUFFICIENT",
                "dominant_narrative": "Could not determine",
                "credibility_score": 0.5,
                "reasoning": f"API error: {exc}",
            }
