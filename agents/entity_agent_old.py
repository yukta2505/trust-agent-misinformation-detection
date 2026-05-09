# """Entity Analysis Agent — checks entity AND context consistency."""

# from __future__ import annotations

# import json
# import logging
# from typing import Any, Dict, List

# from backend.config import Config
# from backend.llm_client import chat_with_fallback
# from backend.utils import clean_text

# LOG = logging.getLogger(__name__)

# SYSTEM_PROMPT = """You are an Entity Analysis Agent for a misinformation detection system.

# Your job: Check whether the IMAGE (described by caption) matches the CLAIM across FOUR dimensions:

# 1. PEOPLE — Do the people in the image match who the claim says?
#    - Claim says "government officials" but image shows students → MISMATCH
#    - Claim says "protesters" but image shows religious ceremony → MISMATCH
#    - Claim says "journalists" but image shows photographers at NASA → MATCH

# 2. PLACE — Does the location in the image match the claim?
#    - Claim says "Tokyo" but image shows Paris/European setting → MISMATCH
#    - Claim says "Baltic Sea" but image shows UK bridge (Gateshead Millennium Bridge) → MISMATCH
#    - Claim says "Kolkata Salt Lake" but image shows a stadium → MISMATCH
#    - Claim says "Mosul/Iraq" and image shows Iraqi flags → MATCH

# 3. EVENT TYPE — Does what is happening in the image match the claim?
#    - Claim says "gas explosion" but image shows military airstrike → MISMATCH
#    - Claim says "emergency government meeting" but image shows theater/Kennedy Center → MISMATCH
#    - Claim says "human rights protest" but image shows crucifixion reenactment → MISMATCH
#    - Claim says "road accident" but image shows building collapse/airstrike → MISMATCH
#    - Claim says "political rally" but image shows Holi festival/color celebration → MISMATCH
#    - Claim says "graduation ceremony" but image shows something else → CHECK

# 4. ORGANISATION — Do organisations/flags/symbols match?
#    - Iraqi flags present, claim says Iraq/Mosul → MATCH
#    - NASA rocket visible, claim says missile test → partial match

# SCORING:
# - 0.9–1.0 → all dimensions match
# - 0.6–0.8 → most match, minor ambiguity
# - 0.3–0.5 → one clear mismatch found
# - 0.0–0.2 → multiple mismatches OR one major mismatch (wrong event type, wrong people, wrong place)

# CRITICAL: Event type mismatch is the strongest signal. If the fundamental nature of what 
# is happening in the image does not match the claim, score 0.0–0.2 regardless of whether 
# location names partially match.

# You must respond ONLY with valid JSON — no markdown fences, no extra text.

# {
#   "claim_entities": {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "EVENT_TYPE": ""},
#   "caption_entities": {"PERSON": [], "ORG": [], "GPE": [], "DATE": [], "EVENT_TYPE": ""},
#   "matches": ["list of matching elements"],
#   "contradictions": ["list of specific contradictions found"],
#   "mismatch_dimensions": ["PEOPLE", "PLACE", "EVENT_TYPE", "ORGANISATION"],
#   "entity_score": 0.0,
#   "reasoning": "specific explanation of what matches and what contradicts"
# }
# """


# class EntityAnalysisAgent:
#     def __init__(self, config: Config) -> None:
#         self.config = config

#     def analyse(
#         self,
#         claim: str,
#         caption: str,
#         evidence_items: List[Dict[str, Any]],
#     ) -> Dict[str, Any]:

#         evidence_summary = "\n".join(
#             f"- [{i+1}] {e.get('title','')}: {e.get('snippet','')}"
#             for i, e in enumerate(evidence_items[:5])
#         )

#         messages = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": f"""Check all four dimensions (PEOPLE, PLACE, EVENT_TYPE, ORGANISATION):

# CLAIM: {clean_text(claim)}

# IMAGE CAPTION: {clean_text(caption)}

# EVIDENCE:
# {evidence_summary or "(no external evidence retrieved)"}

# ANALYSIS CHECKLIST:
# 1. Who does the claim say is in the image? Who is actually shown in the caption?
# 2. Where does the claim say this happened? Where does the caption/evidence suggest?
# 3. What type of event does the claim describe? What type of event does the caption show?
# 4. Are there any flags, logos, or symbols that confirm or contradict the claim?

# Be specific about contradictions. Return JSON only."""},
#         ]

#         try:
#             raw = chat_with_fallback(self.config, messages, max_tokens=700)
#             result = json.loads(raw)
#             LOG.info(
#                 "Entity Agent score: %s | contradictions: %s",
#                 result.get("entity_score"),
#                 result.get("contradictions", []),
#             )
#             return result
#         except Exception as exc:
#             LOG.error("Entity Agent error: %s", exc)
#             return {
#                 "claim_entities": {}, "caption_entities": {},
#                 "matches": [], "contradictions": [],
#                 "mismatch_dimensions": [],
#                 "entity_score": 0.5,
#                 "reasoning": f"Error: {exc}",
#             }


