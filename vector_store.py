"""
vector_store.py
---------------
ChromaDB vector store – document storage and similarity retrieval.

Responsibilities:
  - Create / persist a ChromaDB collection
  - Add pre-chunked documents with metadata
  - Run similarity (nearest-neighbour) search against a query
  - Expose a helper to count stored documents
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChromaDB client + collection (lazy singleton)
# ---------------------------------------------------------------------------
_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """Return (and cache) the ChromaDB collection."""
    global _chroma_client, _collection

    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

        # Use a SentenceTransformer embedding function so we never need an
        # OpenAI key just for embeddings.
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )

        _collection = _chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d docs)",
            config.CHROMA_COLLECTION,
            _collection.count(),
        )

    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def add_documents(
    documents: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
) -> None:
    """
    Insert pre-chunked documents into ChromaDB.

    Parameters
    ----------
    documents : list[str]
        The text content of each chunk.
    metadatas : list[dict]
        Per-chunk metadata (e.g. source filename, chunk index).
    ids : list[str]
        Unique, stable identifiers for each chunk.
    """
    collection = _get_collection()

    # Upsert so re-ingesting the same document doesn't duplicate entries
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.info("Upserted %d document chunks into ChromaDB", len(documents))


def similarity_search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Find the most relevant document chunks for a query.

    Parameters
    ----------
    query : str
        The user's question used as the search vector.
    top_k : int, optional
        Number of results to return.  Defaults to config.TOP_K_DOCS.

    Returns
    -------
    list[dict]
        Each dict contains ``text``, ``metadata``, ``id``, and ``distance``.
    """
    collection = _get_collection()
    k = top_k or config.TOP_K_DOCS

    if collection.count() == 0:
        logger.debug("ChromaDB is empty – no documents to retrieve.")
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB's nested lists (one list per query)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]

    formatted = [
        {
            "text": doc,
            "metadata": meta,
            "id": doc_id,
            "distance": dist,
        }
        for doc, meta, doc_id, dist in zip(docs, metas, ids, distances)
    ]

    logger.debug("ChromaDB returned %d chunks for query", len(formatted))
    return formatted


def count_documents() -> int:
    """Return the total number of chunks stored in ChromaDB."""
    return _get_collection().count()


def delete_collection() -> None:
    """Drop and recreate the ChromaDB collection (for testing / reset)."""
    global _collection
    if _chroma_client:
        _chroma_client.delete_collection(config.CHROMA_COLLECTION)
        _collection = None
        logger.info("ChromaDB collection '%s' deleted", config.CHROMA_COLLECTION)
