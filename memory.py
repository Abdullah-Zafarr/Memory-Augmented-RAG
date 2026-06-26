"""
memory.py
---------
Persistent memory layer powered by Mem0.
"""

from __future__ import annotations

import logging
from typing import Any, List, Dict, Optional

from mem0 import Memory

import config

logger = logging.getLogger(__name__)

def _build_mem0_config() -> Dict[str, Any]:
    if not config.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not configured. Mem0 distillation may fail.")
    return {
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": config.MEM0_COLLECTION,
                "path": "./mem0_db",
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {"model": config.EMBEDDING_MODEL},
        },
        "llm": {
            "provider": "groq",
            "config": {
                "model": config.GROQ_MODEL,
                "api_key": config.GROQ_API_KEY,
                "temperature": 0.1,
                "max_tokens": 512,
            },
        },
    }

_memory_client: Optional[Memory] = None

def get_memory_client() -> Memory:
    global _memory_client
    if _memory_client is None:
        cfg = _build_mem0_config()
        _memory_client = Memory.from_config(cfg)
        logger.info("Mem0 memory client initialised (forced local configuration)")
    return _memory_client

def store_memory(user_id: str, query: str, answer: str) -> None:
    client = get_memory_client()
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    try:
        client.add(messages, user_id=user_id)
        logger.debug("Memory stored for user '%s'", user_id)
    except Exception as exc:
        logger.warning("Failed to store memory for user '%s': %s", user_id, exc)

def retrieve_memory(user_id: str, query: str) -> List[str]:
    client = get_memory_client()
    try:
        results = client.search(
            query=query,
            user_id=user_id,
            limit=config.TOP_K_MEMORY,
        )
        memories: List[str] = []
        for item in results:
            if isinstance(item, dict):
                text = item.get("memory") or item.get("text") or str(item)
            else:
                text = str(item)
            if text:
                memories.append(text)
        logger.debug("Retrieved %d memories for user '%s'", len(memories), user_id)
        return memories
    except Exception as exc:
        logger.warning("Memory retrieval failed for user '%s': %s", user_id, exc)
        return []

def get_all_memories(user_id: str) -> List[Dict[str, Any]]:
    client = get_memory_client()
    try:
        result = client.get_all(user_id=user_id)
        if isinstance(result, dict) and "results" in result:
            return result["results"]
        return result if isinstance(result, list) else []
    except Exception as exc:
        logger.warning("get_all_memories failed for user '%s': %s", user_id, exc)
        return []

def delete_memory(user_id: str, memory_id: str) -> None:
    """Delete a single memory item by its ID."""
    client = get_memory_client()
    try:
        client.delete(memory_id)
        logger.info("Deleted memory '%s' for user '%s'", memory_id, user_id)
    except Exception as exc:
        logger.warning("delete_memory failed for memory '%s': %s", memory_id, exc)

def clear_memory(user_id: str) -> None:
    client = get_memory_client()
    try:
        client.delete_all(user_id=user_id)
        logger.info("Cleared all memories for user '%s'", user_id)
    except Exception as exc:
        logger.warning("clear_memory failed for user '%s': %s", user_id, exc)
