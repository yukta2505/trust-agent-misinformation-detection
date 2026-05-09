"""
Graph node functions for TRUST-AGENT LangGraph pipeline.

Each function:
  - receives the full AgentState
  - does ONE job
  - returns a dict of only the keys it changed

Nodes designed for parallel execution:
  - node_entity_agent
  - node_temporal_agent
  - node_credibility_agent
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List


from agents.aggregator_agent import AggregatorAgent
from agents.credibility_agent import SourceCredibilityAgent
from agents.entity_agent import EntityAnalysisAgent
from agents.plausibility_agent import ClaimPlausibilityAgent
from agents.temporal_agent import TemporalReasoningAgent
from backend.captioning2 import ImageCaptioner
from agents.clip_agent import CLIPVisualAgent


from backend.config import Config
from backend.entity_extraction import EntityExtractor
from backend.evidence_retrieval import ReverseImageSearcher, WebSearcher
from backend.metadata_extractor import MetadataExtractor
from historical_index import HistoricalEvidenceIndex
from graph_state import AgentState
from backend.evidence_filter import filter_relevant_evidence

LOG = logging.getLogger(__name__)

clip_agent = CLIPVisualAgent()


def make_nodes(config: Config) -> Dict[str, Any]:
    """
    Instantiate all heavy objects once and return a dict of
    bound node functions ready to register into the graph.

    This avoids reloading BLIP / spaCy on every call.
    """
    captioner        = ImageCaptioner(config)
    extractor        = EntityExtractor(config)
    meta_extractor   = MetadataExtractor()
    rev_searcher     = ReverseImageSearcher(config)
    web_searcher     = WebSearcher(config)
    historical_index = HistoricalEvidenceIndex(config)
    entity_agent        = EntityAnalysisAgent(config)
    temporal_agent      = TemporalReasoningAgent(config)
    cred_agent          = SourceCredibilityAgent(config)
    plausibility_agent  = ClaimPlausibilityAgent(config)
    aggregator          = AggregatorAgent(config)

    # ── Node 1: Caption the image with BLIP ──────────────────────────────────
    def node_caption(state: AgentState) -> Dict[str, Any]:
        LOG.info("[node] Generating caption for %s", state["image_path"])
        errors: List[str] = list(state.get("errors") or [])
        try:
            caption = captioner.caption(state["image_path"])
            LOG.info("[node] Caption: %s", caption)
            return {"caption": caption}
        except Exception as exc:
            msg = f"Captioning failed: {exc}"
            LOG.warning(msg)
            errors.append(msg)
            return {"caption": "", "errors": errors}

    # ── Node 2: Extract entities with spaCy + metadata from EXIF/text ─────────
    
    def node_extract_entities(state):
        """Extract entities + metadata + run local CLIP analysis."""
        LOG = __import__('logging').getLogger(__name__)
        errors = list(state.get("errors") or [])
 
        combined = f"{state.get('claim', '')} {state.get('caption', '')}"
        entities = extractor.extract(combined)
        LOG.info("[node] Entities: %s", {k: v for k, v in entities.items() if v})
 
        metadata = meta_extractor.extract_all(
            image_path=state.get("image_path", ""),
            claim=state.get("claim", ""),
            caption=state.get("caption", ""),
        )
        LOG.info("[node] Metadata: %s", metadata.get("summary"))
 
        # ── Run CLIP agent locally (no API) ──────────────────────────────────────
        clip_result = {"clip_score": 0.5, "interpretation": "SKIPPED", "reasoning": "CLIP disabled"}
        if config.use_clip_agent:
            try:
                LOG.info("[node] Running local CLIP visual consistency check...")
                clip_result = clip_agent.analyse(
                    image_path=state.get("image_path", ""),
                    claim=state.get("claim", ""),
                    caption=state.get("caption", ""),
                )
                LOG.info(
                    "[node] CLIP: score=%.3f interpretation=%s",
                    clip_result.get("clip_score", 0.5),
                    clip_result.get("interpretation", "?"),
                )
            except Exception as exc:
                msg = f"CLIP agent failed: {exc}"
                LOG.warning(msg)
                errors.append(msg)
                clip_result = {"clip_score": 0.5, "interpretation": "ERROR", "reasoning": msg}
 
        return {
            "entities":    entities,
            "metadata":    metadata,
            "clip_result": clip_result,
            "errors":      errors,
        }
 

    # ── Node 3: Retrieve evidence (reverse image + web) ──────────────────────

    def node_retrieve_evidence(state: AgentState) -> Dict[str, Any]:
        LOG.info("[node] Retrieving evidence")
        errors: List[str] = list(state.get("errors") or [])
        evidence: List[Dict[str, Any]] = []
        top_k = state.get("top_k", 5)

        try:
            rev = rev_searcher.search(
                image_path=state["image_path"], top_k=top_k
            )
            evidence.extend(rev)
            LOG.info("[node] Reverse image: %d results", len(rev))
        except Exception as exc:
            msg = f"Reverse image search failed: {exc}"
            LOG.warning(msg)
            errors.append(msg)

        try:
            web = web_searcher.search(
                claim=state.get("claim", ""),
                entities=state.get("entities", {}),
                caption=state.get("caption", ""),
                top_k=top_k,
            )
            evidence.extend(web)
            LOG.info("[node] Web search: %d results", len(web))
        except Exception as exc:
            msg = f"Web search failed: {exc}"
            LOG.warning(msg)
            errors.append(msg)

        # Historical semantic search (only if index is built)
        try:
            combined = f"{state.get('claim', '')} {state.get('caption', '')}"
            hist = historical_index.search(query=combined, top_k=top_k)
            evidence.extend(hist)
            if hist:
                LOG.info("[node] Historical index: %d results", len(hist))
        except Exception as exc:
            msg = f"Historical index search failed: {exc}"
            LOG.warning(msg)
            errors.append(msg)

        LOG.info("[node] Total evidence before filtering: %d items", len(evidence))

        # ── Filter irrelevant evidence ─────────────────────────────────────────
        # IMPORTANT: this must be BEFORE the return statement
        try:
            from backend.evidence_filter import filter_relevant_evidence
            evidence = filter_relevant_evidence(
                evidence=evidence,
                claim=state.get("claim", ""),
                caption=state.get("caption", ""),
                min_overlap=1,
            )
            LOG.info("[node] After filtering: %d relevant items", len(evidence))
        except Exception as exc:
            LOG.warning("Evidence filtering failed: %s", exc)

        # ── Return filtered evidence ───────────────────────────────────────────
        return {"evidence": evidence, "errors": errors}
    

    # def node_retrieve_evidence(state: AgentState) -> Dict[str, Any]:
    #     LOG.info("[node] Retrieving evidence")
    #     errors: List[str] = list(state.get("errors") or [])
    #     evidence: List[Dict[str, Any]] = []
    #     top_k = state.get("top_k", 5)

    #     try:
    #         rev = rev_searcher.search(
    #             image_path=state["image_path"], top_k=top_k
    #         )
    #         evidence.extend(rev)
    #         LOG.info("[node] Reverse image: %d results", len(rev))
    #     except Exception as exc:
    #         msg = f"Reverse image search failed: {exc}"
    #         LOG.warning(msg)
    #         errors.append(msg)

    #     try:
    #         web = web_searcher.search(
    #             claim=state.get("claim", ""),
    #             entities=state.get("entities", {}),
    #             caption=state.get("caption", ""),
    #             top_k=top_k,
    #         )
    #         evidence.extend(web)
    #         LOG.info("[node] Web search: %d results", len(web))
    #     except Exception as exc:
    #         msg = f"Web search failed: {exc}"
    #         LOG.warning(msg)
    #         errors.append(msg)


    #     # Historical semantic search (only if index is built)
    #     try:
    #         combined = f"{state.get('claim', '')} {state.get('caption', '')}"
    #         hist = historical_index.search(query=combined, top_k=top_k)
    #         evidence.extend(hist)
    #         if hist:
    #             LOG.info("[node] Historical index: %d results", len(hist))
    #     except Exception as exc:
    #         msg = f"Historical index search failed: {exc}"
    #         LOG.warning(msg)
    #         errors.append(msg)
    #     LOG.info("[node] Total evidence: %d items", len(evidence))
 
    #     return {"evidence": evidence, "errors": errors}

        


    # ── Node 4a: Entity Analysis Agent (Claude API) ───────────────────────────
    def node_entity_agent(state: AgentState) -> Dict[str, Any]:
        LOG.info("[node] Entity Analysis Agent")
        errors: List[str] = list(state.get("errors") or [])
        try:
            result = entity_agent.analyse(
                claim=state.get("claim", ""),
                caption=state.get("caption", ""),
                evidence_items=state.get("evidence", []),
            )
            return {"entity_result": result}
        except Exception as exc:
            msg = f"Entity agent failed: {exc}"
            LOG.error(msg)
            errors.append(msg)
            return {
                "entity_result": {"entity_score": 0.5, "reasoning": msg},
                "errors": errors,
            }

    # ── Node 4b: Temporal Reasoning Agent (OpenAI API) ────────────────────────
    def node_temporal_agent(state: AgentState) -> Dict[str, Any]:
        LOG.info("[node] Temporal Reasoning Agent")
        errors: List[str] = list(state.get("errors") or [])
        try:
            # Enrich evidence with metadata summary for richer temporal reasoning
            meta_summary = (state.get("metadata") or {}).get("summary", "")
            result = temporal_agent.analyse(
                claim=state.get("claim", ""),
                caption=state.get("caption", ""),
                evidence_items=state.get("evidence", []),
                metadata_summary=meta_summary,
            )
            return {"temporal_result": result}
        except Exception as exc:
            msg = f"Temporal agent failed: {exc}"
            LOG.error(msg)
            errors.append(msg)
            return {
                "temporal_result": {"temporal_score": 0.5, "reasoning": msg},
                "errors": errors,
            }

    # ── Node 4c: Source Credibility Agent (Claude API) ────────────────────────
    def node_credibility_agent(state: AgentState) -> Dict[str, Any]:
        LOG.info("[node] Source Credibility Agent")
        errors: List[str] = list(state.get("errors") or [])
        try:
            result = cred_agent.analyse(
                claim=state.get("claim", ""),
                caption=state.get("caption", ""),
                evidence_items=state.get("evidence", []),
            )
            return {"credibility_result": result}
        except Exception as exc:
            msg = f"Credibility agent failed: {exc}"
            LOG.error(msg)
            errors.append(msg)
            return {
                "credibility_result": {"credibility_score": 0.5, "reasoning": msg},
                "errors": errors,
            }

    # ── Node 5: Claim Plausibility Agent (soft OOC signal) ──────────────────
    def node_plausibility_agent(state: AgentState) -> Dict[str, Any]:
        LOG.info("[node] Claim Plausibility Agent")
        errors: List[str] = list(state.get("errors") or [])
        try:
            result = plausibility_agent.analyse(
                claim=state.get("claim", ""),
                caption=state.get("caption", ""),
            )
            return {"plausibility_result": result}
        except Exception as exc:
            msg = f"Plausibility agent failed: {exc}"
            LOG.error(msg)
            errors.append(msg)
            return {
                "plausibility_result": {
                    "plausibility_score": 0.70,
                    "red_flags": [],
                    "positive_signals": [],
                    "claim_type_assessment": "ordinary",
                    "reasoning": msg,
                },
                "errors": errors,
            }

    # ── Node 6: Aggregator — Claude generates the human explanation ───────────
    def node_aggregator(state: AgentState) -> Dict[str, Any]:
        LOG.info("[node] Aggregator Agent")
        errors: List[str] = list(state.get("errors") or [])
        try:
            agg = aggregator.aggregate(
                claim=state.get("claim", ""),
                caption=state.get("caption", ""),
                entity_result=state.get("entity_result", {}),
                temporal_result=state.get("temporal_result", {}),
                credibility_result=state.get("credibility_result", {}),
                clip_result=state.get("clip_result", {}),
                evidence_items=state.get("evidence", []),
                plausibility_result=state.get("plausibility_result", {}),
            )
            return {
                "verdict": agg.get("verdict", state.get("math_verdict")),
                "confidence_percent": int(agg.get("confidence_percent", 50)),
                "explanation": agg.get("explanation", ""),
                "key_evidence_for_verdict": agg.get("key_evidence_for_verdict", []),
                "flags": agg.get("flags", []),
                "threshold_used": float(agg.get("threshold_used", 0.0)),
                "plausibility_score": float(agg.get("plausibility_score", 0.70)),
                "ooc_signal_count": int(agg.get("ooc_signal_count", 0)),
                "ooc_category": agg.get("ooc_category", "none"),
            }
        except Exception as exc:
            msg = f"Aggregator failed: {exc}"
            LOG.error(msg)
            errors.append(msg)
            fs = state.get("final_score", 0.5)
            mv = state.get("math_verdict", "OUT-OF-CONTEXT")
            return {
                "verdict": mv,
                "confidence_percent": int(abs(fs - 0.5) * 200),
                "explanation": "Analysis completed with partial results.",
                "key_evidence_for_verdict": [],
                "flags": [],
                "errors": errors,
            }

    return {
        "node_caption":           node_caption,
        "node_extract_entities":  node_extract_entities,
        "node_retrieve_evidence": node_retrieve_evidence,
        "node_entity_agent":      node_entity_agent,
        "node_temporal_agent":    node_temporal_agent,
        "node_credibility_agent": node_credibility_agent,
        "node_plausibility_agent": node_plausibility_agent,
        "node_aggregator":        node_aggregator,
    }
