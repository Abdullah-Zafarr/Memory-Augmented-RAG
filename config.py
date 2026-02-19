"""
config.py
---------
Centralised configuration for the Memory-Augmented-RAG system.
All settings are loaded from environment variables (via .env) so that no
secrets are hard-coded in source files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM (Groq)
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ---------------------------------------------------------------------------
# Mem0 memory
# ---------------------------------------------------------------------------
MEM0_API_KEY: str = os.getenv("MEM0_API_KEY", "")          # optional cloud key
MEM0_COLLECTION: str = os.getenv("MEM0_COLLECTION", "rag_memory")
MEM0_USER_ID: str = os.getenv("MEM0_USER_ID", "default_user")

# ---------------------------------------------------------------------------
# ChromaDB vector store
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "rag_docs")

# ---------------------------------------------------------------------------
# Embedding model (used by ChromaDB)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# Document ingestion
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
TOP_K_DOCS: int = int(os.getenv("TOP_K_DOCS", 5))          # docs from ChromaDB
TOP_K_MEMORY: int = int(os.getenv("TOP_K_MEMORY", 3))      # entries from Mem0

# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", 1024))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", 0.3))

# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------
APP_TITLE: str = "Memory-Augmented RAG"
APP_ICON: str = "🧠"
DEFAULT_USER_ID: str = "default_user"
