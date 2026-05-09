"""Source Credibility Agent — evaluates trustworthiness of evidence sources."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from backend.config import Config
from backend.llm_client import chat_with_fallback
from backend.utils import clean_text

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Source Credibility Agent for a misinformation detection system.

Your job: Evaluate how trustworthy the retrieved evidence sources are and whether 
they support or contradict the claim.

════════════════════════════════════════════════════════
MOST IMPORTANT RULE — READ FIRST:

If NO evidence was retrieved OR all sources are irrelevant:
  → credibility_score MUST be 0.55 (neutral)
  → cross_source_agreement MUST be "INSUFFICIENT"
  → Do NOT score below 0.55 just because there is no evidence
  → Absence of evidence is NOT proof of misinformation

This is critical. Scoring 0.2 when there is no evidence is WRONG.
0.2 means credible sources ACTIVELY CONTRADICTED the claim.
If no sources exist, that is neutral (0.55), not suspicious.
════════════════════════════════════════════════════════

SCORING GUIDE:
- 0.80–1.00 → Multiple HIGH-credibility sources confirm the claim
- 0.65–0.79 → Some credible sources support the claim
- 0.55       → No relevant sources found (NEUTRAL — use this as default)
- 0.35–0.54 → Mixed signals, some doubt from credible sources
- 0.00–0.34 → Credible sources SPECIFICALLY AND ACTIVELY contradict the claim

Credibility tiers:
  HIGH   : BBC, Reuters, AP, AFP, NYT, Washington Post, Al Jazeera, .gov, NASA
  MEDIUM : Regional news, Wikipedia, established outlets
  LOW    : Social media posts, unknown blogs, tabloids
  UNKNOWN: No identifiable source

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISCAPTIONED CONTENT — SPECIAL RULE:
Miscaptioned content is when a REAL image from event A is paired with a claim
describing event B. Key signals:
  - The source evidence is about a DIFFERENT event than the claim
  - The claim has specific, verifiable named entities (person, place, org)
  - Evidence directly names a DIFFERENT entity for the same image
  → In this case: credibility_score 0.15 to 0.35 is appropriate
  → Only applies when evidence is DIRECTLY contradictory (not merely different topic)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You must respond ONLY with valid JSON — no markdown fences, no extra text.

Response schema:
{
  "sources_evaluated": [
    {
      "source": "name or domain",
      "credibility_tier": "HIGH|MEDIUM|LOW|UNKNOWN",
      "supports_claim": true,
      "reason": "one-line rationale"
    }
  ],
  "cross_source_agreement": "AGREE|DISAGREE|MIXED|INSUFFICIENT",
  "dominant_narrative": "what credible sources say, or 'No relevant sources found'",
  "credibility_score": 0.55,
  "reasoning": "explanation — if no sources, explicitly state neutral default"
}"""


class SourceCredibilityAgent:
    def __init__(self, config: Config) -> None:
        self.config = config

    def analyse(
        self,
        claim: str,
        caption: str,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        # If no evidence at all — return neutral immediately without API call
        if not evidence_items:
            LOG.info("Credibility Agent: no evidence — returning neutral 0.55")
            return {
                "sources_evaluated": [],
                "cross_source_agreement": "INSUFFICIENT",
                "dominant_narrative": "No relevant sources found",
                "credibility_score": 0.55,
                "reasoning": "No evidence retrieved — neutral default applied",
            }

        source_lines = []
        for i, e in enumerate(evidence_items[:6]):
            source = e.get("source") or e.get("url") or "unknown"
            source_lines.append(
                f"[{i+1}] Source: {source}\n"
                f"    Title: {e.get('title', '')}\n"
                f"    Snippet: {e.get('snippet', '')}"
            )
        source_block = "\n".join(source_lines)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Evaluate source credibility for:

CLAIM: {clean_text(claim)}

IMAGE CAPTION: {clean_text(caption)}

RETRIEVED SOURCES:
{source_block}

REMINDER: 
- If sources are irrelevant to the claim → credibility_score = 0.55 (neutral)
- Only score BELOW 0.55 if a credible source ACTIVELY CONTRADICTS the claim
- Miscaptioned detection: if evidence is about a DIFFERENT named entity than claimed → score 0.20-0.35
Return JSON only."""},
        ]

        try:
            raw = chat_with_fallback(self.config, messages, max_tokens=700)
            result = json.loads(raw)

            score = float(result.get("credibility_score", 0.55))
            agreement = result.get("cross_source_agreement", "INSUFFICIENT")
            sources = result.get("sources_evaluated", [])

            active_contradictions = [
                s for s in sources
                if s.get("supports_claim") is False
                and s.get("credibility_tier") in ("HIGH", "MEDIUM")
            ]

            # ── Hard safety overrides ────────────────────────────────────────
            # Rule 1: INSUFFICIENT agreement with no active contradictions → neutral
            if agreement == "INSUFFICIENT" and not active_contradictions:
                if score < 0.50:
                    LOG.warning(
                        "Credibility: agreement=INSUFFICIENT, no contradictions "
                        "but score=%.2f — overriding to 0.55", score
                    )
                    result["credibility_score"] = 0.55
                    result["reasoning"] = (
                        result.get("reasoning", "") +
                        " [Override: INSUFFICIENT with no active contradictions → neutral 0.55]"
                    )

            # Rule 2: Very low score but no HIGH/MEDIUM contradictions → neutral
            score = float(result.get("credibility_score", 0.55))
            if score < 0.35 and not active_contradictions:
                LOG.warning(
                    "Credibility: score=%.2f with no active HIGH/MEDIUM contradictions "
                    "— overriding to 0.55", score
                )
                result["credibility_score"] = 0.55
                result["reasoning"] = (
                    result.get("reasoning", "") +
                    " [Override: No credible source actively contradicted claim → neutral 0.55]"
                )

            LOG.info(
                "Credibility Agent: score=%.3f | agreement=%s | sources=%d",
                result.get("credibility_score"),
                result.get("cross_source_agreement"),
                len(result.get("sources_evaluated", [])),
            )
            return result

        except Exception as exc:
            LOG.error("Credibility Agent error: %s", exc)
            return {
                "sources_evaluated": [],
                "cross_source_agreement": "INSUFFICIENT",
                "dominant_narrative": "Could not determine",
                "credibility_score": 0.55,
                "reasoning": f"Error — neutral default: {exc}",
            }