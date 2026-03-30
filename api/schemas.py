"""Pydantic request and response models for the TRUST-AGENT API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class AnalyseResponse(BaseModel):
    """Full response returned by POST /analyse."""

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict: str                          # "PRISTINE" or "OUT-OF-CONTEXT"
    confidence_percent: int               # 0 – 100
    explanation: str                      # plain-English reason

    # ── Image understanding ───────────────────────────────────────────────────
    caption: str                          # BLIP-generated caption

    # ── Agent scores ──────────────────────────────────────────────────────────
    entity_score: float                   # 0.0 – 1.0
    temporal_score: float
    credibility_score: float
    final_score: float                    # weighted fusion

    # ── Supporting detail ──────────────────────────────────────────────────────
    key_evidence_for_verdict: List[str]   # bullet points from aggregator
    flags: List[str]                      # red flags (empty if PRISTINE)
    evidence: List[Dict[str, Any]]        # ranked evidence items

    # ── Meta ──────────────────────────────────────────────────────────────────
    processing_time_sec: float
    errors: List[str]                     # non-fatal warnings


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str = "1.0.0"
