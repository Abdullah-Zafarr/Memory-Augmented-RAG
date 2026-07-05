from typing import List, Dict, Any

def calculate_chunk_stats(chunks: List[str]) -> Dict[str, Any]:
    """Calculate character-level statistics for a list of document chunks.

    Args:
        chunks: A list of strings representing the document chunks.
    Returns:
        A dict containing count, average, max, and min character lengths.
    """
    if not chunks:
        return {"count": 0, "avg_length": 0, "max_length": 0, "min_length": 0}
    
    lengths = [len(c) for c in chunks]
    return {
        "count": len(chunks),
        "avg_length": sum(lengths) / len(chunks),
        "max_length": max(lengths),
        "min_length": min(lengths)
    }
