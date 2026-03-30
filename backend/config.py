"""Central configuration for TRUST-AGENT pipeline."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parents[1] / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass


@dataclass
class Config:
    # ── LLM (OpenAI) ──────────────────────────────────────────────────────────
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # ── Vision / NLP models ───────────────────────────────────────────────────
    blip_model: str = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-large")
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    sentence_model: str = os.getenv(
        "SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── External APIs ─────────────────────────────────────────────────────────
    serpapi_key: str = field(default_factory=lambda: os.getenv("SERPAPI_API_KEY", ""))
    newsapi_key: str = field(default_factory=lambda: os.getenv("NEWSAPI_KEY", ""))
    imgbb_api_key: str = field(default_factory=lambda: os.getenv("IMGBB_API_KEY", ""))

    # ── Storage / artifacts ───────────────────────────────────────────────────
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "artifacts")
    historical_index_path: str = os.getenv(
        "HISTORICAL_INDEX_PATH", "artifacts/historical.faiss"
    )
    historical_meta_path: str = os.getenv(
        "HISTORICAL_META_PATH", "artifacts/historical_meta.json"
    )

    # ── Pipeline defaults ─────────────────────────────────────────────────────
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "20"))

    # ── Confidence scoring weights (must sum to 1.0) ──────────────────────────
    weight_entity: float = float(os.getenv("WEIGHT_ENTITY", "0.35"))
    weight_temporal: float = float(os.getenv("WEIGHT_TEMPORAL", "0.35"))
    weight_credibility: float = float(os.getenv("WEIGHT_CREDIBILITY", "0.30"))

    # Threshold above which verdict is PRISTINE
    pristine_threshold: float = float(os.getenv("PRISTINE_THRESHOLD", "0.60"))
