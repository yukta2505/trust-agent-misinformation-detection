"""
LLM client factory for TRUST-AGENT.

All four agents call this instead of directly creating OpenAI().
Switching provider = change one line in .env (LLM_PROVIDER=groq).

Supported providers:
  openai  → OpenAI GPT-4o / GPT-4o-mini
  groq    → Groq LLaMA 3.3 70B (free tier, fast)
  ollama  → Local Ollama (fully offline, free)
"""

from __future__ import annotations

from backend.config import Config


def get_llm_client(config: Config):
    """
    Return an OpenAI-compatible client for the configured provider.

    All three providers (OpenAI, Groq, Ollama) use the same
    OpenAI Python SDK — only the base_url and api_key differ.
    """
    from openai import OpenAI

    if config.llm_provider == "groq":
        if not config.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
            )
        return OpenAI(
            api_key=config.groq_api_key,
            base_url=config.groq_base_url,
        )

    if config.llm_provider == "ollama":
        # Ollama runs locally — no API key needed
        return OpenAI(
            api_key="ollama",
            base_url=config.ollama_base_url,
        )

    # Default: OpenAI
    if not config.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set in .env file."
        )
    return OpenAI(api_key=config.openai_api_key)