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
    # ── LLM Provider ──────────────────────────────────────────────────────────
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # ── Groq (free) ───────────────────────────────────────────────────────────
    groq_api_key: str = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY", "")
    )
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # ── Ollama (fully local) ──────────────────────────────────────────────────
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

    # ── Local models ──────────────────────────────────────────────────────────
    blip_model: str = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-large")
    clip_model: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    sentence_model: str = os.getenv(
        "SENTENCE_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── External APIs ─────────────────────────────────────────────────────────
    serpapi_key: str = field(default_factory=lambda: os.getenv("SERPAPI_API_KEY", ""))
    newsapi_key: str = field(default_factory=lambda: os.getenv("NEWSAPI_KEY", ""))
    imgbb_api_key: str = field(default_factory=lambda: os.getenv("IMGBB_API_KEY", ""))

    # ── Storage ───────────────────────────────────────────────────────────────
    artifacts_dir: str = os.getenv("ARTIFACTS_DIR", "artifacts")
    historical_index_path: str = os.getenv(
        "HISTORICAL_INDEX_PATH", "artifacts/historical.faiss"
    )
    historical_meta_path: str = os.getenv(
        "HISTORICAL_META_PATH", "artifacts/historical_meta.json"
    )

    # ── Pipeline ──────────────────────────────────────────────────────────────
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "45"))
    pristine_threshold: float = float(os.getenv("PRISTINE_THRESHOLD", "0.60"))

    # ── Agent score weights ───────────────────────────────────────────────────
    # With CLIP (5 agents, weights must sum to 1.0):
    weight_entity:      float = float(os.getenv("WEIGHT_ENTITY",      "0.25"))
    weight_temporal:    float = float(os.getenv("WEIGHT_TEMPORAL",    "0.30"))
    weight_credibility: float = float(os.getenv("WEIGHT_CREDIBILITY", "0.20"))
    weight_clip:        float = float(os.getenv("WEIGHT_CLIP",        "0.25"))
    # Without CLIP (3 agents, old weights — used as fallback):
    weight_entity_no_clip:      float = 0.35
    weight_temporal_no_clip:    float = 0.35
    weight_credibility_no_clip: float = 0.30

    # ── Feature flags ─────────────────────────────────────────────────────────
    # Set USE_CLIP=false in .env to disable CLIP agent
    use_clip_agent: bool = os.getenv("USE_CLIP", "true").lower() == "true"

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def active_model(self) -> str:
        if self.llm_provider == "groq":
            return self.groq_model
        if self.llm_provider == "ollama":
            return self.ollama_model
        return self.openai_model

    @property
    def active_api_key(self) -> str:
        if self.llm_provider == "groq":
            return self.groq_api_key
        if self.llm_provider == "ollama":
            return "ollama"
        return self.openai_api_key

    @property
    def active_base_url(self):
        if self.llm_provider == "groq":
            return self.groq_base_url
        if self.llm_provider == "ollama":
            return self.ollama_base_url
        return None

    @property
    def is_offline_capable(self) -> bool:
        """True if the pipeline can run without any API calls."""
        return (
            self.llm_provider == "ollama"
            and self.use_clip_agent
        )