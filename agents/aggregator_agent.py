"""
Aggregator Agent v2 — full adaptive verdict with 5-signal fusion.

Improvements over v1:
  1. Integrates ClaimPlausibilityAgent as 5th soft signal
  2. Credibility guard: neutralises bad credibility scores when no evidence
  3. Adaptive threshold: 0.60 → 0.56 → 0.52 based on OOC signal count
  4. Absolute OOC overrides for confirmed TYPE_A and miscaptioning
  5. Better weight redistribution when credibility evidence is absent
  6. ooc_category classification (temporal_trick / miscaptioned / false_context)

Weight scheme:
  entity      : 0.35
  temporal    : 0.30
  credibility : 0.20  (redistributed to e/t/clip when no evidence)
  clip        : 0.15
  plausibility: soft signal only (adjusts threshold, not raw score)

Thresholds:
  Standard    : 0.60  (0 or 1 OOC signals)
  Moderate    : 0.56  (2 OOC signals)
  Low         : 0.52  (3+ OOC signals)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.config import Config
from backend.llm_client import chat_with_fallback
from backend.utils import clean_text

LOG = logging.getLogger(__name__)

_W_ENTITY      = 0.35
_W_TEMPORAL    = 0.30
_W_CREDIBILITY = 0.20
_W_CLIP        = 0.15

_THRESHOLD_STANDARD = 0.60
_THRESHOLD_MODERATE = 0.56
_THRESHOLD_LOW      = 0.52

AGGREGATOR_PROMPT = """You are the final verdict agent for a misinformation detection system.

VERDICT LABELS:
- PRISTINE: image and claim are consistent — no evidence of manipulation
- OUT-OF-CONTEXT: real image used with false/misleading claim

You must respond ONLY with valid JSON — no markdown fences.

Response schema:
{
  "explanation": "2-3 sentence plain English explanation for the public",
  "confidence_percent": 65,
  "key_evidence_for_verdict": ["up to 3 most important signals"],
  "flags": ["specific red flags — empty list if PRISTINE"],
  "ooc_category": "temporal_trick|miscaptioned|false_context|none"
}"""


def _cred_is_absent(cred: Dict) -> bool:
    return (
        cred.get("cross_source_agreement") == "INSUFFICIENT"
        or cred.get("dominant_narrative") == "No relevant sources found"
        or len(cred.get("sources_evaluated", [])) == 0
    )


def compute_final_score(
    e_score: float, t_score: float, c_score: float, clip_score: float,
    cred_result: Dict, config: Config,
) -> float:
    we = float(getattr(config, "weight_entity",      None) or _W_ENTITY)
    wt = float(getattr(config, "weight_temporal",    None) or _W_TEMPORAL)
    wc = float(getattr(config, "weight_credibility", None) or _W_CREDIBILITY)
    wl = float(getattr(config, "weight_clip",        None) or _W_CLIP)

    if _cred_is_absent(cred_result):
        c_score = max(c_score, 0.55)
        we += wc * 0.40
        wt += wc * 0.35
        wl += wc * 0.25
        wc = 0.0

    total = we + wt + wc + wl
    score = (we/total)*e_score + (wt/total)*t_score + (wc/total)*c_score + (wl/total)*clip_score
    LOG.debug("Score: e=%.3f t=%.3f c=%.3f clip=%.3f → %.4f", e_score, t_score, c_score, clip_score, score)
    return score


def count_signals(
    ent: Dict, tmp: Dict, cred: Dict, clip_score: float,
    plausibility: Optional[Dict] = None,
) -> Tuple[int, List[str]]:
    sigs: List[str] = []

    ctype  = tmp.get("claim_type", "TYPE_C")
    tscore = float(tmp.get("temporal_score", 0.65))
    if ctype == "TYPE_A":
        sigs.append(f"Temporal TYPE_A time trick (score={tscore:.2f})")
    elif ctype == "TYPE_B_WRONG":
        sigs.append(f"Temporal wrong date claimed (score={tscore:.2f})")
    elif tscore < 0.40:
        sigs.append(f"Temporal inconsistency (score={tscore:.2f})")

    escore = float(ent.get("entity_score", 0.65))
    miscs  = ent.get("miscaptioned_signals", [])
    contrs = ent.get("contradictions", [])
    if miscs:
        sigs.append(f"Miscaptioning signal: {miscs[0][:80]}")
    elif contrs:
        sigs.append(f"Entity contradiction: {contrs[0][:80]}")
    elif escore < 0.35:
        sigs.append(f"Low entity score ({escore:.2f})")

    if not _cred_is_absent(cred):
        cscore = float(cred.get("credibility_score", 0.55))
        agree  = cred.get("cross_source_agreement", "")
        if agree == "DISAGREE" and cscore < 0.40:
            sigs.append(f"Credible sources disagree (score={cscore:.2f})")
        elif cscore < 0.35:
            sigs.append(f"Low credibility from evidence ({cscore:.2f})")

    if clip_score < 0.40:
        sigs.append(f"Visual inconsistency CLIP={clip_score:.2f}")

    if plausibility:
        pscore = float(plausibility.get("plausibility_score", 0.70))
        pflags = plausibility.get("red_flags", [])
        if pscore < 0.45 or (pscore < 0.60 and pflags):
            sigs.append(
                f"Implausible claim (plausibility={pscore:.2f})"
                + (f": {pflags[0][:60]}" if pflags else "")
            )

    return len(sigs), sigs


def determine_verdict(
    score: float,
    ent: Dict, tmp: Dict, cred: Dict,
    clip_score: float,
    plausibility: Optional[Dict] = None,
) -> Tuple[str, float]:

    nsigs, sigs = count_signals(ent, tmp, cred, clip_score, plausibility)

    if nsigs >= 3:
        threshold = _THRESHOLD_LOW
    elif nsigs == 2:
        threshold = _THRESHOLD_MODERATE
    else:
        threshold = _THRESHOLD_STANDARD

    LOG.info("OOC signals=%d → threshold=%.2f | %s", nsigs, threshold, sigs)

    # Absolute overrides
    ctype  = tmp.get("claim_type", "TYPE_C")
    tscore = float(tmp.get("temporal_score", 0.65))
    escore = float(ent.get("entity_score", 0.65))
    miscs  = ent.get("miscaptioned_signals", [])

    if ctype == "TYPE_A" and tscore <= 0.20:
        return "OUT-OF-CONTEXT", threshold
    if miscs and escore <= 0.25:
        return "OUT-OF-CONTEXT", threshold

    return ("PRISTINE" if score >= threshold else "OUT-OF-CONTEXT"), threshold


class AggregatorAgent:
    def __init__(self, config: Config) -> None:
        self.config = config

    def aggregate(
        self,
        claim: str,
        caption: str,
        entity_result: Dict,
        temporal_result: Dict,
        credibility_result: Dict,
        clip_result: Dict,
        evidence_items: List[Dict],
        plausibility_result: Optional[Dict] = None,
    ) -> Dict[str, Any]:

        e_score    = float(entity_result.get("entity_score", 0.65))
        t_score    = float(temporal_result.get("temporal_score", 0.65))
        c_score    = float(credibility_result.get("credibility_score", 0.55))
        clip_score = float(clip_result.get("clip_score", 0.55))

        final_score = compute_final_score(
            e_score, t_score, c_score, clip_score,
            credibility_result, self.config,
        )

        verdict, threshold = determine_verdict(
            final_score, entity_result, temporal_result,
            credibility_result, clip_score, plausibility_result,
        )

        LOG.info("Verdict=%s score=%.4f threshold=%.2f", verdict, final_score, threshold)

        nsigs, sigs = count_signals(
            entity_result, temporal_result, credibility_result,
            clip_score, plausibility_result,
        )
        p_score = float(plausibility_result.get("plausibility_score", 0.70)) if plausibility_result else 0.70

        agent_block = (
            f"Entity:       {e_score:.3f} — {entity_result.get('reasoning','')[:100]}\n"
            f"Temporal:     {t_score:.3f} — {temporal_result.get('claim_type','')} — {temporal_result.get('reasoning','')[:100]}\n"
            f"Credibility:  {c_score:.3f} — {credibility_result.get('cross_source_agreement','')} — {credibility_result.get('dominant_narrative','')[:100]}\n"
            f"CLIP:         {clip_score:.3f} — {clip_result.get('interpretation','')}\n"
            f"Plausibility: {p_score:.3f} — {(plausibility_result or {}).get('reasoning','N/A')[:100]}\n"
            f"Final Score:  {final_score:.4f} (threshold={threshold:.2f})\n"
            f"OOC Signals:  {nsigs} — {'; '.join(sigs[:3])}\n"
            f"VERDICT:      {verdict}"
        )

        top_ev = "\n".join(
            f"- {e.get('source','')}: {e.get('title','')[:70]}"
            for e in evidence_items[:3]
        ) or "(no evidence retrieved)"

        messages = [
            {"role": "system", "content": AGGREGATOR_PROMPT},
            {"role": "user", "content": (
                f"CLAIM: {clean_text(claim)}\n"
                f"IMAGE CAPTION: {clean_text(caption)}\n\n"
                f"AGENT RESULTS:\n{agent_block}\n\n"
                f"TOP EVIDENCE:\n{top_ev}\n\n"
                "Return JSON only."
            )},
        ]

        try:
            raw = chat_with_fallback(self.config, messages, max_tokens=600)
            result = json.loads(raw)
        except Exception as exc:
            LOG.error("Aggregator LLM error: %s", exc)
            confidence = min(99, max(30, int(abs(final_score - threshold) * 300 + 50)))
            result = {
                "explanation": f"System verdict: {verdict}. Score: {final_score:.3f}.",
                "confidence_percent": confidence,
                "key_evidence_for_verdict": sigs[:3],
                "flags": sigs[:3] if verdict == "OUT-OF-CONTEXT" else [],
                "ooc_category": "none",
            }

        result.update({
            "verdict":            verdict,
            "final_score":        round(final_score, 4),
            "threshold_used":     threshold,
            "entity_score":       e_score,
            "temporal_score":     t_score,
            "credibility_score":  c_score,
            "clip_score":         clip_score,
            "plausibility_score": p_score,
            "ooc_signal_count":   nsigs,
        })
        return result