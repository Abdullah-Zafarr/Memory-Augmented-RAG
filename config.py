import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

g = lambda k, d: os.getenv(k, d)
gi = lambda k, d: int(g(k, d))
gf = lambda k, d: float(g(k, d))

GROQ_API_KEY = g("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    logger.warning("GROQ_API_KEY is not set in environment variables.")

GROQ_MODEL = g("GROQ_MODEL", "llama-3.3-70b-versatile")
MEM0_API_KEY = g("MEM0_API_KEY", "")
MEM0_COLLECTION = g("MEM0_COLLECTION", "rag_memory")
MEM0_USER_ID = g("MEM0_USER_ID", "default_user")
CHROMA_PERSIST_DIR = g("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION = g("CHROMA_COLLECTION", "rag_docs")
EMBEDDING_MODEL = g("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = gi("CHUNK_SIZE", 800)
CHUNK_OVERLAP = gi("CHUNK_OVERLAP", 100)
TOP_K_DOCS = gi("TOP_K_DOCS", 5)
TOP_K_MEMORY = gi("TOP_K_MEMORY", 3)
MAX_TOKENS = gi("MAX_TOKENS", 1024)
TEMPERATURE = gf("TEMPERATURE", 0.3)
APP_TITLE = "Memory-Augmented RAG"
APP_ICON = "🧠"
DEFAULT_USER_ID = "default_user"
