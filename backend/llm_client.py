"""
LLM client factory with smart rate-limit handling.

Behaviour on 429 (Groq rate limit):
  1. Parse the actual wait time from Groq's error message
  2. Wait exactly that long (e.g. "Please try again in 4m24s" → wait 264s)
  3. Retry the same call
  4. If still failing → fall back to OpenAI
  5. If OpenAI also missing → raise with clear message

This ensures benchmark_eval.py never crashes on Groq 429.
It just waits and continues automatically.
"""

from __future__ import annotations

import logging
import re
import time

LOG = logging.getLogger(__name__)


# ── Provider client factory ────────────────────────────────────────────────────

def get_llm_client(config):
    """Return an OpenAI-compatible client for the configured provider."""
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


# ── Wait time parser ───────────────────────────────────────────────────────────

def _parse_wait_seconds(error_message: str) -> int:
    """
    Parse the wait time from Groq's 429 error message.

    Examples:
      "Please try again in 4m24.383s"  → 264
      "Please try again in 15m17.568s" → 917
      "Please try again in 30s"         → 30
      (no match)                        → 60 (default)
    """
    # Try "Xm Y.Zs" format
    m = re.search(r"(\d+)m(\d+(?:\.\d+)?)s", error_message)
    if m:
        minutes  = int(m.group(1))
        seconds  = float(m.group(2))
        total    = int(minutes * 60 + seconds) + 5  # add 5s buffer
        return total

    # Try plain "Xs" format
    m = re.search(r"in (\d+(?:\.\d+)?)s", error_message)
    if m:
        return int(float(m.group(1))) + 5

    return 60  # default: wait 60 seconds


# ── Main function used by all agents ──────────────────────────────────────────

def chat_with_fallback(
    config,
    messages: list,
    max_tokens: int = 800,
    max_rate_limit_waits: int = 3,
) -> str:
    """
    Call the configured LLM.

    On Groq 429:
      - Parses the exact wait time from the error message
      - Waits that long (can be several minutes)
      - Retries up to max_rate_limit_waits times
      - Falls back to OpenAI if still failing

    Parameters
    ----------
    config                 : Config object
    messages               : chat messages list
    max_tokens             : max response tokens
    max_rate_limit_waits   : how many times to wait-and-retry on 429
                             before falling back to OpenAI
    """
    from openai import OpenAI, RateLimitError

    client = get_llm_client(config)
    model  = config.active_model

    LOG.debug("LLM call: provider=%s model=%s", config.llm_provider, model)

    rate_limit_waits = 0

    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content.strip()

        except RateLimitError as exc:
            err_str = str(exc)
            wait_sec = _parse_wait_seconds(err_str)

            if rate_limit_waits < max_rate_limit_waits:
                rate_limit_waits += 1
                LOG.warning(
                    "Groq rate limit (attempt %d/%d). "
                    "Waiting %ds before retry... | %s",
                    rate_limit_waits, max_rate_limit_waits,
                    wait_sec, err_str[:120],
                )
                # Show countdown every 30 seconds
                elapsed = 0
                while elapsed < wait_sec:
                    chunk = min(30, wait_sec - elapsed)
                    time.sleep(chunk)
                    elapsed += chunk
                    remaining = wait_sec - elapsed
                    if remaining > 0:
                        LOG.info(
                            "Still waiting for rate limit reset... %ds remaining",
                            remaining,
                        )
                LOG.info("Rate limit wait done. Retrying...")
                continue  # retry the same call

            else:
                # Exhausted retries — fall back to OpenAI
                LOG.warning(
                    "Groq rate limit exhausted after %d waits. "
                    "Falling back to OpenAI...",
                    max_rate_limit_waits,
                )
                break

        except Exception as exc:
            LOG.error("LLM call failed: %s", exc)
            raise

    # ── OpenAI fallback ───────────────────────────────────────────────────────
    if config.openai_api_key and config.llm_provider != "openai":
        try:
            LOG.info("Using OpenAI fallback: %s", config.openai_model)
            fallback = OpenAI(api_key=config.openai_api_key)
            response = fallback.chat.completions.create(
                model=config.openai_model,
                max_tokens=max_tokens,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            raise RuntimeError(
                "Both Groq and OpenAI are rate-limited. "
                "Please wait and run again with --skip-done."
            )
        except Exception as exc:
            LOG.error("OpenAI fallback failed: %s", exc)
            raise

    raise RuntimeError(
        f"LLM provider '{config.llm_provider}' rate-limited and no fallback available. "
        f"Set OPENAI_API_KEY in .env as fallback, or wait for Groq reset and "
        f"run again with --skip-done flag."
    )