"""Shared application dependencies for FastAPI."""

from __future__ import annotations

from functools import lru_cache

from backend.config import Config
from orchestrator import TrustAgentOrchestrator


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()


@lru_cache(maxsize=1)
def get_orchestrator() -> TrustAgentOrchestrator:
    return TrustAgentOrchestrator(get_config())
