"""
llm.py
------
Thin wrapper around the Groq API.

Responsibilities:
  - Initialise the Groq client once
  - Expose a `generate()` function that accepts a list of messages
    and returns the assistant's text response
  - Centralise error handling and retry logic for LLM calls
"""

from __future__ import annotations

import logging
from typing import Any

from groq import Groq

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client (singleton-ish – module-level import creates it once)
# ---------------------------------------------------------------------------
_client: Groq | None = None


def _get_client() -> Groq:
    """Return a cached Groq client, creating it on first call."""
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file and restart the app."
            )
        _client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("Groq client initialised with model %s", config.GROQ_MODEL)
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """
    Send a list of chat messages to Groq and return the assistant reply.

    Parameters
    ----------
    messages:
        OpenAI-compatible message list, e.g.
        [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    model:
        Override the default model from config.
    max_tokens:
        Override the default max_tokens from config.
    temperature:
        Override the default temperature from config.

    Returns
    -------
    str
        The assistant's text response.

    Raises
    ------
    groq.APIError
        Propagated on API-level failures (rate limits, auth issues, etc.)
    """
    client = _get_client()

    _model = model or config.GROQ_MODEL
    _max_tokens = max_tokens or config.MAX_TOKENS
    _temperature = temperature if temperature is not None else config.TEMPERATURE

    logger.debug(
        "Calling Groq | model=%s | messages=%d | max_tokens=%d | temp=%.2f",
        _model,
        len(messages),
        _max_tokens,
        _temperature,
    )

    response = client.chat.completions.create(
        model=_model,
        messages=messages,          # type: ignore[arg-type]
        max_tokens=_max_tokens,
        temperature=_temperature,
    )

    reply: str = response.choices[0].message.content or ""
    logger.debug("Groq reply (%d chars)", len(reply))
    return reply


def build_messages(
    system_prompt: str,
    user_message: str,
) -> list[dict[str, str]]:
    """
    Helper: construct a minimal messages list for a single-turn exchange.

    Parameters
    ----------
    system_prompt : str
        Instructions / context injected as the system message.
    user_message : str
        The raw user query.

    Returns
    -------
    list[dict[str, str]]
        A two-element messages list ready for `generate()`.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
