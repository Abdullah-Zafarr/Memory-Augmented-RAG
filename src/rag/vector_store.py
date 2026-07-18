"""
vector_store.py
---------------
ChromaDB vector store – document storage and similarity retrieval.
"""

from __future__ import annotations

import logging
from typing import Any, List, Dict, Optional

import chromadb
from chromadb.utils import embedding_functions

from src.core import config

logger = logging.getLogger(__name__)

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None

def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection

    if _collection is None:
        try:
            import os
            os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config.EMBEDDING_MODEL
            )
            _collection = _chroma_client.get_or_create_collection(
                name=config.CHROMA_COLLECTION,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' ready with %d documents",
                config.CHROMA_COLLECTION,
                _collection.count(),
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {e}")
            raise e

    return _collection

def add_documents(
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
) -> None:
    try:
        collection = _get_collection()
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        logger.info("Upserted %d document chunks into ChromaDB", len(documents))
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB: {e}")
        raise e

def similarity_search(
    query: str, 
    top_k: Optional[int] = None, 
    filter_dict: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    collection = _get_collection()
    k = top_k or config.TOP_K_DOCS

    if collection.count() == 0:
        logger.debug("ChromaDB is empty – no documents to retrieve.")
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(k, collection.count()),
            where=filter_dict,
            include=["documents", "metadatas", "distances"],
        )
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
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return []

def count_documents() -> int:
    try:
        return _get_collection().count()
    except Exception as e:
        logger.error(f"Failed to count documents: {e}")
        return 0

def delete_collection() -> None:
    global _collection
    if _chroma_client:
        try:
            _chroma_client.delete_collection(config.CHROMA_COLLECTION)
            _collection = None
            logger.info("ChromaDB collection '%s' deleted", config.CHROMA_COLLECTION)
        except Exception as e:
            logger.error(f"Failed to delete ChromaDB collection: {e}")
