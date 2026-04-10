"""Source Credibility Agent — evaluates evidence sources."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from backend.config import Config
from backend.llm_client import chat_with_fallback
from backend.utils import clean_text

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Source Credibility Agent for a misinformation detection system.

Your job: Evaluate whether the retrieved evidence SUPPORTS or CONTRADICTS the claim,
and how credible the sources are.

IMPORTANT RULES:
- No evidence / irrelevant evidence → credibility_score: 0.55 (neutral)
- Credible sources CONFIRM the claim → score 0.75–1.0
- Credible sources CONTRADICT the claim → score 0.0–0.3
- Mixed signals → score 0.4–0.6

CONTEXT TRICK detection:
If the claim describes a specific event (e.g. "gas explosion", "protest", "meeting")
but credible sources show the image is actually from a DIFFERENT type of event
(e.g. "airstrike", "festival", "theater performance"), that is strong evidence of 
context manipulation → score LOW (0.1–0.3)

HIGH credibility: BBC, Reuters, AP, AFP, NYT, Al Jazeera, Washington Post, .gov sites
MEDIUM: Regional news, Wikipedia, established outlets  
LOW: Social media, unknown blogs, tabloids, YouTube channels

You must respond ONLY with valid JSON — no markdown fences.

{
  "sources_evaluated": [
    {
      "source": "source name",
      "credibility_tier": "HIGH|MEDIUM|LOW|UNKNOWN",
      "supports_claim": true or false or null,
      "actual_event_described": "what this source says the image actually shows",
      "reason": "one line"
    }
  ],
  "cross_source_agreement": "AGREE|DISAGREE|MIXED|INSUFFICIENT",
  "dominant_narrative": "what credible sources actually say about this image/event",
  "context_mismatch_detected": false,
  "credibility_score": 0.0,
  "reasoning": "overall assessment"
}
"""


class SourceCredibilityAgent:
    def __init__(self, config: Config) -> None:
        self.config = config

    def analyse(
        self,
        claim: str,
        caption: str,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        source_lines = []
        for i, e in enumerate(evidence_items[:6]):
            source = e.get("source") or e.get("url") or "unknown"
            source_lines.append(
                f"[{i+1}] Source: {source}\n"
                f"    Title: {e.get('title','')}\n"
                f"    Snippet: {e.get('snippet','')}"
            )
        source_block = "\n".join(source_lines) or "(no sources retrieved)"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Evaluate source credibility:

CLAIM: {clean_text(claim)}
IMAGE CAPTION: {clean_text(caption)}

RETRIEVED SOURCES:
{source_block}

Key questions:
1. Do any sources confirm what the claim says?
2. Do any sources reveal what the image ACTUALLY shows (different from claim)?
3. Does evidence suggest a context mismatch (wrong event type, wrong location)?

Return JSON only."""},
        ]

        try:
            raw = chat_with_fallback(self.config, messages, max_tokens=800)
            result = json.loads(raw)
            LOG.info(
                "Credibility Agent score: %s | context_mismatch: %s",
                result.get("credibility_score"),
                result.get("context_mismatch_detected"),
            )
            return result
        except Exception as exc:
            LOG.error("Credibility Agent error: %s", exc)
            return {
                "sources_evaluated": [],
                "cross_source_agreement": "INSUFFICIENT",
                "dominant_narrative": "Could not determine",
                "context_mismatch_detected": False,
                "credibility_score": 0.55,
                "reasoning": f"Error: {exc}",
            }