from src.core.utils import calculate_chunk_stats

def test_calculate_chunk_stats_empty():
    stats = calculate_chunk_stats([])
    assert stats["count"] == 0
    assert stats["avg_length"] == 0

def test_calculate_chunk_stats_valid():
    stats = calculate_chunk_stats(["hello", "world!"])
    assert stats["count"] == 2
    assert stats["avg_length"] == 5.5
    assert stats["max_length"] == 6
    assert stats["min_length"] == 5
