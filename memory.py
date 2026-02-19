"""
memory.py
---------
Persistent memory layer powered by Mem0.

Mem0 provides long-term, user-scoped memory storage with built-in semantic
search.  This module exposes a clean interface so the rest of the codebase
never touches the Mem0 SDK directly.

Key responsibilities:
  - Initialise the Mem0 Memory client (cloud or local)
  - Store a new interaction (query + answer) for a given user
  - Retrieve the most relevant past memories for a query
  - (Optional) clear / reset a user's memory
"""

from __future__ import annotations

import logging
from typing import Any

from mem0 import Memory

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------
def _build_mem0_config() -> dict[str, Any]:
    """
    Build the Mem0 configuration dictionary.

    If MEM0_API_KEY is provided, we delegate to the Mem0 cloud backend.
    Otherwise, we run fully locally – Mem0 will use its default local
    vector + graph setup, which requires no extra credentials.
    """
    # To avoid "OPENAI_API_KEY" errors with Mem0 Cloud (which defaults to OpenAI),
    # we will force Local Mode usage which uses our Groq configuration.
    # if config.MEM0_API_KEY:
    #     # Cloud mode – no local setup required
    #     return {"api_key": config.MEM0_API_KEY}

    # Local mode – instruct Mem0 to use a local ChromaDB backend
    # so that memory persists across sessions without any cloud account.
    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": config.MEM0_COLLECTION,
                "path": "./mem0_db",          # separate from the RAG chroma_db
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": config.EMBEDDING_MODEL},
        },
        "llm": {
            # Mem0 uses an LLM internally for memory distillation
            "provider": "groq",
            "config": {
                "model": config.GROQ_MODEL,
                "api_key": config.GROQ_API_KEY,
                "temperature": 0.1,
                "max_tokens": 512,
            },
        },
    }


# Singleton client – created lazily on first use
_memory_client: Memory | None = None


def get_memory_client() -> Memory:
    """Return (and cache) the Mem0 Memory client."""
    global _memory_client
    if _memory_client is None:
        cfg = _build_mem0_config()
        # Always use from_config with our local settings (Groq/Chroma)
        # to ensure we don't accidentally trigger OpenAI dependencies.
        _memory_client = Memory.from_config(cfg)
        logger.info("Mem0 memory client initialised (forced local configuration)")
    return _memory_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def store_memory(user_id: str, query: str, answer: str) -> None:
    """
    Persist an interaction (query + assistant answer) in Mem0.

    Mem0 automatically distils the conversation into atomic memory facts so
    that future retrievals are semantically precise rather than verbatim.

    Parameters
    ----------
    user_id : str
        Unique identifier for the user (enables per-user memory isolation).
    query : str
        The user's original question.
    answer : str
        The assistant's generated answer.
    """
    client = get_memory_client()

    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]

    try:
        client.add(messages, user_id=user_id)
        logger.debug("Memory stored for user '%s'", user_id)
    except Exception as exc:
        # Non-fatal – a memory write failure should not crash the pipeline
        logger.warning("Failed to store memory for user '%s': %s", user_id, exc)


def retrieve_memory(user_id: str, query: str) -> list[str]:
    """
    Retrieve the most relevant memories for a given query and user.

    Parameters
    ----------
    user_id : str
        The user whose memory to search.
    query : str
        The current user question (used as the search vector).

    Returns
    -------
    list[str]
        A list of plain-text memory strings ordered by relevance.
        Returns an empty list if no memory exists yet.
    """
    client = get_memory_client()

    try:
        results = client.search(
            query=query,
            user_id=user_id,
            limit=config.TOP_K_MEMORY,
        )

        # Mem0 returns a list of dicts; extract the memory text
        memories: list[str] = []
        for item in results:
            if isinstance(item, dict):
                text = item.get("memory") or item.get("text") or str(item)
            else:
                text = str(item)
            if text:
                memories.append(text)

        logger.debug(
            "Retrieved %d memories for user '%s'", len(memories), user_id
        )
        return memories

    except Exception as exc:
        logger.warning("Memory retrieval failed for user '%s': %s", user_id, exc)
        return []


def get_all_memories(user_id: str) -> list[dict[str, Any]]:
    """
    Return all stored memory entries for a user (for display in the UI).

    Parameters
    ----------
    user_id : str

    Returns
    -------
    list[dict]
        Raw Mem0 memory objects.
    """
    client = get_memory_client()
    try:
        result = client.get_all(user_id=user_id)
        # mem0 >= 0.1.x wraps results in {"results": [...]}
        if isinstance(result, dict) and "results" in result:
            return result["results"]
        return result if isinstance(result, list) else []
    except Exception as exc:
        logger.warning("get_all_memories failed for user '%s': %s", user_id, exc)
        return []


def clear_memory(user_id: str) -> None:
    """
    Delete all stored memories for a given user.

    Parameters
    ----------
    user_id : str
    """
    client = get_memory_client()
    try:
        client.delete_all(user_id=user_id)
        logger.info("Cleared all memories for user '%s'", user_id)
    except Exception as exc:
        logger.warning("clear_memory failed for user '%s': %s", user_id, exc)
