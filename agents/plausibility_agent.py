"""
Claim Plausibility Agent — 5th independent signal for TRUST-AGENT.

Purpose:
  Detects suspicious claims that SOUND like misinformation even when
  external evidence is missing. Works entirely from claim text + caption.

Why this is needed:
  The existing 4 agents default to "neutral" when no evidence is found.
  For miscaptioned content, evidence is often absent (the real image source
  is not indexed, or the claim is too generic to retrieve evidence for).
  This agent provides a claim-level sanity check independent of evidence.

What it detects:
  1. EXTRAORDINARY CLAIMS — "world's first selfie", "reveals moon landing hoax"
  2. WRONG ENTITY TYPE — claim describes a person, caption shows an object
  3. SPECIFIC VERIFIABLE CLAIMS — very specific but unverifiable assertions
  4. VAGUE CLAIM / VIVID IMAGE MISMATCH — generic claim for a very specific scene
  5. CONSISTENT CLAIM — claim matches what caption describes, nothing suspicious

Score:
  0.80–1.00 → Claim is plausible, nothing suspicious
  0.60–0.79 → Claim is mostly plausible, minor questions
  0.40–0.59 → Claim has suspicious elements worth flagging
  0.00–0.39 → Claim is implausible, contradicted by caption, or extraordinary

Integration:
  This score is used in the aggregator as a soft OOC signal.
  It does NOT have a direct weight in the final_score formula.
  Instead it adjusts the adaptive threshold when suspicious.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from backend.config import Config
from backend.llm_client import chat_with_fallback
from backend.utils import clean_text

LOG = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Claim Plausibility Agent for a misinformation detection system.

Your job: Evaluate whether the CLAIM TEXT itself sounds plausible and consistent
with the IMAGE CAPTION — without any external evidence.

You are NOT checking if the claim is true. You are checking if it SOUNDS suspicious.

WHAT TO LOOK FOR:

RED FLAGS (lower the score):
  1. Extraordinary claims: "world's first selfie", "proof that X never happened", 
     "secret photo reveals conspiracy", "banned photograph"
  2. Entity type mismatch: claim says PERSON but caption describes OBJECT/SCENE
  3. Over-specific unverifiable claim: very precise details (names, dates, locations)
     with no way to verify from the caption
  4. Claim contradicts obvious visual: caption describes something clearly different
  5. Claim adds suspicious context: "taken during [unrelated famous event]",
     "shows [person] doing [unexpected thing]"
  6. False framing: "image shows X" but caption doesn't support X at all

POSITIVE SIGNALS (raise the score):
  1. Claim matches what caption shows (same type of scene)
  2. Claim is generic/neutral ("image shows protesters")
  3. Claim mentions art, art installations, science, nature (usually true)
  4. Caption strongly confirms the visual described in the claim

IMPORTANT RULES:
  - Do NOT penalise historical facts just for being old
  - Do NOT penalise claims mentioning past events with specific dates
  - Only flag as suspicious if the claim ADDS questionable context not in the image
  - A claim about a movie still, artwork, or 3D render is suspicious if
    it's presented as real news

You must respond ONLY with valid JSON — no markdown fences.

Response schema:
{
  "plausibility_score": 0.75,
  "red_flags": ["list any suspicious elements — empty if none"],
  "positive_signals": ["list supporting elements"],
  "claim_type_assessment": "ordinary|extraordinary|misleading_framing|entity_mismatch",
  "reasoning": "brief explanation"
}

plausibility_score guide:
  0.80–1.00 → Plausible. Claim matches caption. Nothing suspicious.
  0.60–0.79 → Mostly plausible, minor uncertainty.
  0.40–0.59 → Suspicious elements present.
  0.00–0.39 → High suspicion: extraordinary claims, clear mismatches.
"""


class ClaimPlausibilityAgent:
    """
    5th agent — evaluates claim plausibility purely from text + caption.
    No external evidence required.
    Score is used as a soft OOC signal in the aggregator threshold logic.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def analyse(self, claim: str, caption: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""CLAIM: {clean_text(claim)}

IMAGE CAPTION: {clean_text(caption)}

Assess the plausibility of this claim given the caption.
Does the claim match the caption? Does it make extraordinary assertions?
Return JSON only."""},
        ]

        try:
            raw = chat_with_fallback(self.config, messages, max_tokens=400)
            result = json.loads(raw)
            score = float(result.get("plausibility_score", 0.70))

            LOG.info(
                "Plausibility Agent: score=%.3f type=%s flags=%s",
                score,
                result.get("claim_type_assessment", ""),
                result.get("red_flags", []),
            )
            return result

        except Exception as exc:
            LOG.error("Plausibility Agent error: %s", exc)
            return {
                "plausibility_score": 0.70,
                "red_flags": [],
                "positive_signals": [],
                "claim_type_assessment": "ordinary",
                "reasoning": f"Error — neutral default: {exc}",
            }