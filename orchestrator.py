"""
TRUST-AGENT LangGraph Orchestrator
====================================
Builds a stateful LangGraph pipeline where the three reasoning
agents (entity, temporal, credibility) run in PARALLEL, then
their results are fused and passed to the aggregator.

Graph structure:
                        ┌──────────────┐
                        │   __start__  │
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  caption     │   (BLIP)
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  entities    │   (spaCy)
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  evidence    │   (SerpAPI + web)
                        └──┬───┬───┬──┘
                           │   │   │        ← fan-out (parallel)
              ┌────────────▼┐ ┌▼─────────┐ ┌▼────────────┐
              │entity_agent │ │temp_agent│ │cred_agent   │
              └────────────┬┘ └┬─────────┘ └┬────────────┘
                           │   │             │             ← fan-in
                        ┌──▼───▼─────────────▼──┐
                        │     score_fusion       │
                        └──────────┬─────────────┘
                                   │
                        ┌──────────▼─────────────┐
                        │      aggregator         │   (Claude)
                        └──────────┬─────────────┘
                                   │
                               __end__
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from backend.config import Config
from graph_nodes import make_nodes
from graph_state import AgentState

LOG = logging.getLogger(__name__)


@dataclass
class TrustAgentResult:
    """Clean result object returned from the orchestrator."""

    claim: str
    image_path: str
    caption: str
    entities: Dict[str, List[str]]
    evidence: List[Dict[str, Any]]
    entity_analysis: Dict[str, Any]
    temporal_analysis: Dict[str, Any]
    credibility_analysis: Dict[str, Any]
    entity_score: float
    temporal_score: float
    credibility_score: float
    final_score: float
    verdict: str
    confidence_percent: int
    explanation: str
    key_evidence_for_verdict: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    processing_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "image_path": self.image_path,
            "caption": self.caption,
            "entities": self.entities,
            "evidence": self.evidence,
            "agent_scores": {
                "entity": round(self.entity_score, 4),
                "temporal": round(self.temporal_score, 4),
                "credibility": round(self.credibility_score, 4),
                "final_weighted": round(self.final_score, 4),
            },
            "entity_analysis": self.entity_analysis,
            "temporal_analysis": self.temporal_analysis,
            "credibility_analysis": self.credibility_analysis,
            "verdict": self.verdict,
            "confidence_percent": self.confidence_percent,
            "explanation": self.explanation,
            "key_evidence_for_verdict": self.key_evidence_for_verdict,
            "flags": self.flags,
            "errors": self.errors,
            "processing_time_sec": round(self.processing_time_sec, 2),
        }


class TrustAgentOrchestrator:
    """
    LangGraph-based orchestrator for TRUST-AGENT.

    Usage
    -----
    from orchestrator import TrustAgentOrchestrator
    from backend.config import Config

    orc = TrustAgentOrchestrator(Config())
    result = orc.run(image_path="photo.jpg", claim="This is from 2024.")
    print(result.verdict, result.confidence_percent)
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        nodes = make_nodes(self.config)
        builder = StateGraph(AgentState)

        # Register nodes
        builder.add_node("caption",           nodes["node_caption"])
        builder.add_node("extract_entities",  nodes["node_extract_entities"])
        builder.add_node("retrieve_evidence", nodes["node_retrieve_evidence"])
        builder.add_node("entity_agent",      nodes["node_entity_agent"])
        builder.add_node("temporal_agent",    nodes["node_temporal_agent"])
        builder.add_node("credibility_agent", nodes["node_credibility_agent"])
        builder.add_node("score_fusion",      nodes["node_score_fusion"])
        builder.add_node("aggregator",        nodes["node_aggregator"])

        # Sequential: start → caption → entities → evidence
        builder.add_edge(START,               "caption")
        builder.add_edge("caption",           "extract_entities")
        builder.add_edge("extract_entities",  "retrieve_evidence")

        # Fan-out: evidence → 3 parallel agents
        builder.add_edge("retrieve_evidence", "entity_agent")
        builder.add_edge("retrieve_evidence", "temporal_agent")
        builder.add_edge("retrieve_evidence", "credibility_agent")

        # Fan-in: all 3 → score_fusion
        builder.add_edge("entity_agent",      "score_fusion")
        builder.add_edge("temporal_agent",    "score_fusion")
        builder.add_edge("credibility_agent", "score_fusion")

        # Final: fusion → aggregator → end
        builder.add_edge("score_fusion",      "aggregator")
        builder.add_edge("aggregator",        END)

        return builder.compile()

    def run(
        self,
        image_path: str,
        claim: str,
        top_k: Optional[int] = None,
    ) -> TrustAgentResult:
        """Execute the full TRUST-AGENT pipeline."""
        start = time.time()
        k = top_k or self.config.default_top_k
        LOG.info("=== TRUST-AGENT START | image=%s ===", image_path)

        initial_state: AgentState = {
            "image_path": image_path,
            "claim": claim,
            "top_k": k,
            "errors": [],
        }

        final_state: AgentState = self._graph.invoke(initial_state)

        elapsed = time.time() - start
        LOG.info(
            "=== TRUST-AGENT DONE | verdict=%s | confidence=%s%% | %.1fs ===",
            final_state.get("verdict"),
            final_state.get("confidence_percent"),
            elapsed,
        )

        return TrustAgentResult(
            claim=claim,
            image_path=image_path,
            caption=final_state.get("caption", ""),
            entities=final_state.get("entities", {}),
            evidence=final_state.get("evidence", []),
            entity_analysis=final_state.get("entity_result", {}),
            temporal_analysis=final_state.get("temporal_result", {}),
            credibility_analysis=final_state.get("credibility_result", {}),
            entity_score=final_state.get("entity_score", 0.5),
            temporal_score=final_state.get("temporal_score", 0.5),
            credibility_score=final_state.get("credibility_score", 0.5),
            final_score=final_state.get("final_score", 0.5),
            verdict=final_state.get("verdict", "OUT-OF-CONTEXT"),
            confidence_percent=final_state.get("confidence_percent", 50),
            explanation=final_state.get("explanation", ""),
            key_evidence_for_verdict=final_state.get("key_evidence_for_verdict", []),
            flags=final_state.get("flags", []),
            errors=final_state.get("errors", []),
            processing_time_sec=elapsed,
        )

    def get_graph_diagram(self) -> str:
        """ASCII diagram of the graph for debugging."""
        try:
            return self._graph.get_graph().draw_ascii()
        except Exception:
            return "Graph diagram unavailable."
