"""
LLM client factory with automatic fallback on rate limit.

Priority order:
  1. Configured provider (groq / openai / ollama)
  2. If 429 rate limit → automatically falls back to OpenAI
  3. If OpenAI also fails → returns error with clear message
"""

from __future__ import annotations

import logging
import time
from typing import Any

LOG = logging.getLogger(__name__)


def get_llm_client(config):
    """Return OpenAI-compatible client for configured provider."""
    from openai import OpenAI

    if config.llm_provider == "groq":
        if not config.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get free key at https://console.groq.com"
            )
        return OpenAI(
            api_key=config.groq_api_key,
            base_url=config.groq_base_url,
        )

    if config.llm_provider == "ollama":
        return OpenAI(
            api_key="ollama",
            base_url=config.ollama_base_url,
        )

    # Default: OpenAI
    if not config.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=config.openai_api_key)


def chat_with_fallback(
    config,
    messages: list,
    max_tokens: int = 800,
    retries: int = 2,
) -> str:
    """
    Call the configured LLM with automatic fallback to OpenAI on 429.

    Parameters
    ----------
    config     : Config object
    messages   : list of {"role": ..., "content": ...} dicts
    max_tokens : max response tokens
    retries    : number of retry attempts before falling back

    Returns
    -------
    Response text string
    """
    from openai import OpenAI, RateLimitError

    # ── Try primary provider ──────────────────────────────────────────────────
    client = get_llm_client(config)
    model  = config.active_model

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content.strip()

        except RateLimitError as exc:
            wait_msg = str(exc)
            LOG.warning(
                "Rate limit on %s (attempt %d/%d): %s",
                model, attempt + 1, retries, wait_msg[:100]
            )
            if attempt < retries - 1:
                time.sleep(3)
                continue

            # All retries exhausted — fall back to OpenAI
            LOG.warning("Falling back to OpenAI after Groq rate limit")
            break

        except Exception as exc:
            LOG.error("LLM call failed (attempt %d): %s", attempt + 1, exc)
            if attempt < retries - 1:
                time.sleep(2)
                continue
            raise

    # ── Fallback: OpenAI ──────────────────────────────────────────────────────
    if config.openai_api_key and config.llm_provider != "openai":
        try:
            fallback_client = OpenAI(api_key=config.openai_api_key)
            LOG.info("Using OpenAI fallback with model: %s", config.openai_model)
            response = fallback_client.chat.completions.create(
                model=config.openai_model,
                max_tokens=max_tokens,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            LOG.error("OpenAI fallback also failed: %s", exc)
            raise

    raise RuntimeError(
        f"All LLM providers failed. "
        f"Provider={config.llm_provider}, model={model}. "
        f"If using Groq free tier, daily limit may be reached. "
        f"Wait for reset or switch to OpenAI."
    )