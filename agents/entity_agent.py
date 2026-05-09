"""Entity Analysis Agent — checks entity consistency between image and claim.
Improved: adds miscaptioned detection via evidence cross-reference.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from backend.config import Config
from backend.llm_client import chat_with_fallback
from backend.utils import clean_text

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Entity Analysis Agent for a misinformation detection system.

Your job: Given an image caption, a text claim, and any retrieved evidence,
check whether the entities (people, places, organisations, objects, events)
in the claim are CONSISTENT or CONTRADICTED.

You must check TWO sources of contradiction:
  A) Caption vs Claim — does the image caption contradict the claim's entities?
  B) Evidence vs Claim — do retrieved sources name DIFFERENT entities for this image?

═══════════════════════════════════════════════════════════
MISCAPTIONED DETECTION (most important):

Miscaptioned = a REAL image is paired with a FALSE description.
The image may be authentic but the claimed person/place/event is wrong.

Signals of miscaptioning:
  - Evidence shows the image is of Person X, but claim says Person Y
  - Evidence shows photo was at Location A, but claim says Location B
  - Evidence shows image is from Event X, but claim says Event Y
  - Multiple credible sources consistently attribute image to DIFFERENT entity

If you find evidence-based entity mismatch → entity_score: 0.10 to 0.30
═══════════════════════════════════════════════════════════

CRITICAL DEFAULT RULES:

1. If the claim has NO specific named entities (no person name, no city, no org):
   → entity_score: 0.65 (neutral — nothing to contradict)

2. If the claim has entities BUT neither caption nor evidence can confirm or deny:
   → entity_score: 0.60 (neutral — benefit of doubt)

3. If the caption OR evidence CONFIRMS entities in the claim:
   → entity_score: 0.80 to 1.0

4. If the caption SPECIFICALLY CONTRADICTS a key entity (visual mismatch):
   → entity_score: 0.10 to 0.25
   Example: claim says "Paris", caption says "London buildings visible"

5. If evidence names a DIFFERENT specific entity for the same image:
   → entity_score: 0.10 to 0.30 (miscaptioned signal)

6. NEVER return 0.0 just because you cannot verify an entity.
   0.0 is ONLY for confirmed contradictions — not for missing information.

The absence of evidence is NOT evidence of contradiction.

You must respond ONLY with valid JSON — no markdown fences, no extra text.

Response schema:
{
  "claim_entities": {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "EVENT": []},
  "caption_entities": {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "EVENT": []},
  "evidence_entities": {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "EVENT": []},
  "matches": ["entities appearing in both claim and caption/evidence"],
  "contradictions": ["only SPECIFIC factual conflicts — empty if none found"],
  "miscaptioned_signals": ["evidence-based mismatches, empty if none"],
  "entity_score": 0.0,
  "reasoning": "brief explanation of your scoring decision"
}

entity_score guide:
- 0.80–1.00 → caption/evidence confirms claim entities
- 0.60–0.79 → no contradiction found (neutral/consistent)
- 0.40–0.59 → vague, unverifiable but slight doubt
- 0.10–0.39 → specific contradiction found (visual or evidence-based)
- 0.00–0.09 → severe, multi-source confirmed contradiction

Examples of NEUTRAL scoring (score 0.65):
  Claim: "Image shows electric scooters abandoned by Mobike"
  Caption: "rows of green electric scooters on a street"
  Evidence: (none)
  → No contradiction → score 0.65

Examples of HIGH scoring (score 0.85):
  Claim: "Protesters wave Iraqi flags in Baghdad"
  Caption: "crowd waving Iraqi flags in an urban area"
  → Entities match → score 0.85

Examples of MISCAPTIONED scoring (score 0.20):
  Claim: "Photograph of politician John Smith at rally in 2021"
  Caption: "man speaking at podium"
  Evidence: [Reuters] "Photo shows politician Jane Doe at London conference 2019"
  → Evidence directly contradicts claimed person/date → score 0.20

Examples of VISUAL CONTRADICTION scoring (score 0.10):
  Claim: "Flood in Chennai, India"
  Caption: "flooded streets with European-style buildings and cars"
  → Visual contradiction → score 0.10
"""


class EntityAnalysisAgent:
    def __init__(self, config: Config) -> None:
        self.config = config

    def analyse(
        self,
        claim: str,
        caption: str,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        # Build evidence summary emphasising entity information
        evidence_lines = []
        for i, e in enumerate(evidence_items[:5]):
            evidence_lines.append(
                f"- [{i+1}] Source: {e.get('source', 'unknown')} | "
                f"Title: {e.get('title', '')} | "
                f"Snippet: {e.get('snippet', '')}"
            )
        evidence_summary = (
            "\n".join(evidence_lines)
            or "(no external evidence retrieved — use caption and claim only)"
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""CLAIM: {clean_text(claim)}

IMAGE CAPTION: {clean_text(caption)}

EVIDENCE (retrieved from web):
{evidence_summary}

ANALYSIS INSTRUCTIONS:
1. Extract entities from: claim, caption, AND evidence
2. Check if caption entities match claim entities (visual consistency)
3. IMPORTANT: Check if evidence names DIFFERENT entities than claimed (miscaptioning)
4. Only score below 0.40 for SPECIFIC confirmed contradictions from caption or evidence
5. No evidence ≠ contradiction → score 0.60 or higher if no contradiction found

Return JSON only."""},
        ]

        try:
            raw = chat_with_fallback(self.config, messages, max_tokens=700)
            result = json.loads(raw)

            score = float(result.get("entity_score", 0.65))
            contradictions = result.get("contradictions", [])
            miscaptioned_signals = result.get("miscaptioned_signals", [])

            all_contradictions = contradictions + miscaptioned_signals

            # Safety check — prevent systematic 0.0 scoring with no evidence
            if score < 0.3 and not all_contradictions:
                LOG.warning(
                    "Entity agent returned score=%.2f with no contradictions — "
                    "overriding to 0.60 (neutral)", score
                )
                result["entity_score"] = 0.60
                result["reasoning"] = (
                    result.get("reasoning", "") +
                    " [Score overridden to neutral — no specific contradictions found]"
                )

            LOG.info(
                "Entity Agent score: %.3f | contradictions: %s | miscaptioned: %s",
                result.get("entity_score"),
                result.get("contradictions", []),
                result.get("miscaptioned_signals", []),
            )
            return result

        except Exception as exc:
            LOG.error("Entity Agent error: %s", exc)
            return {
                "claim_entities": {}, "caption_entities": {}, "evidence_entities": {},
                "matches": [], "contradictions": [], "miscaptioned_signals": [],
                "entity_score": 0.65,
                "reasoning": f"Error — defaulting to neutral: {exc}",
            }