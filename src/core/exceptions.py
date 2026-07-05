class RAGException(Exception):
    """Base exception class for Memory-Augmented RAG system."""
    pass

class DatabaseConnectionError(RAGException):
    """Raised when ChromaDB or Mem0 connection fails."""
    pass

class LLMServiceError(RAGException):
    """Raised when Groq API call fails or times out."""
    pass

class MemoryAccessError(RAGException):
    """Raised when retrieving or storing memory fails."""
    pass
