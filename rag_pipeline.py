"""
rag_pipeline.py
---------------
Core application logic orchestration.

Responsibilities:
  - Coordinate retrieval from Vector Store (documents) and Mem0 (user memory).
  - Construct the prompt with context.
  - Call the LLM to generate an answer.
  - Update Mem0 with the new interaction.
  - Handle document ingestion (text chunking -> vector store).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
import llm
import memory
import vector_store

logger = logging.getLogger(__name__)


def ingest_text(text: str, source_name: str) -> int:
    """
    Chunk and ingest a raw text document into the vector store.

    Parameters
    ----------
    text : str
        The raw text content of the document.
    source_name : str
        A label for the document (e.g. filename).

    Returns
    -------
    int
        Number of chunks created and stored.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    if not chunks:
        logger.warning("No chunks generated from document '%s'", source_name)
        return 0

    # Create distinct IDs for each chunk
    chunk_ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

    vector_store.add_documents(documents=chunks, metadatas=metadatas, ids=chunk_ids)
    logger.info(
        "Ingested '%s' (%d chars) -> %d chunks", source_name, len(text), len(chunks)
    )
    return len(chunks)


def query_rag(user_id: str, question: str) -> dict[str, Any]:
    """
    Full RAG flow: Memory + Docs -> LLM -> Answer -> Memory Update.

    Parameters
    ----------
    user_id : str
        The user making the request.
    question : str
        The user's question.

    Returns
    -------
    dict
        {
            "answer": str,
            "retrieved_docs": list[str],
            "retrieved_memories": list[str]
        }
    """
    # 1. Parallel retrieval (conceptual, sequential in practice here)
    #    - Relevant documents from ChromaDB
    #    - Relevant past interactions from Mem0
    docs = vector_store.similarity_search(question, top_k=config.TOP_K_DOCS)
    doc_texts = [d["text"] for d in docs]
    
    memories = memory.retrieve_memory(user_id, question)

    # 2. Construct System Prompt
    system_prompt = _construct_prompt(doc_texts, memories)

    # 3. Call LLM
    messages = llm.build_messages(system_prompt, question)
    answer = llm.generate(messages)

    # 4. Store interaction in Mem0 (fire-and-forget-ish)
    memory.store_memory(user_id, question, answer)

    return {
        "answer": answer,
        "retrieved_docs": doc_texts,
        "retrieved_memories": memories,
    }


def _construct_prompt(doc_texts: list[str], memories: list[str]) -> str:
    """Combine retrieved context into a single system prompt."""
    
    # Format distinct context sections
    doc_context = "\n---\n".join(doc_texts) if doc_texts else "No specific documents found."
    mem_context = "\n- ".join(memories) if memories else "No relevant past memories."

    return f"""You are a helpful assistant with access to a knowledge base and long-term memory.

### Context from Documents:
{doc_context}

### Context from User Memory (Past Interactions):
- {mem_context}

Instructions:
1. Answer the user's question using the provided context.
2. Prioritise the Documents for factual information.
3. Use the Memory to personalise the answer or recall previous preferences/details.
4. If the answer is not in the context, say so, but try to be helpful based on general knowledge if safe.
"""
