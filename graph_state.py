"""
Shared state schema for the TRUST-AGENT LangGraph pipeline.

Every node reads from and writes to this TypedDict.
LangGraph merges partial updates automatically — each node
only needs to return the keys it changed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ── Inputs ────────────────────────────────────────────────────────────────
    image_path: str
    claim: str
    top_k: int

    # ── Preprocessing outputs ─────────────────────────────────────────────────
    caption: str
    entities: Dict[str, List[str]]
    metadata: Dict[str, Any]          # EXIF + claim temporal clues

    # ── Evidence ──────────────────────────────────────────────────────────────
    evidence: List[Dict[str, Any]]

    # ── Agent outputs (written in parallel) ──────────────────────────────────
    entity_result: Dict[str, Any]
    temporal_result: Dict[str, Any]
    credibility_result: Dict[str, Any]

    # ── Scoring ───────────────────────────────────────────────────────────────
    entity_score: float
    temporal_score: float
    credibility_score: float
    final_score: float
    math_verdict: str          # threshold-based: "PRISTINE" or "OUT-OF-CONTEXT"

    # ── Final output (from aggregator agent) ─────────────────────────────────
    verdict: str
    confidence_percent: int
    explanation: str
    key_evidence_for_verdict: List[str]
    flags: List[str]

    # ── Meta ──────────────────────────────────────────────────────────────────
    errors: List[str]          # non-fatal errors collected during the run
    processing_time_sec: float
