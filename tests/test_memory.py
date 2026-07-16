import pytest
from src.rag import memory

def test_memory_retrieval_empty():
    mems = memory.retrieve_memory("non_existent_user_xyz", "hello")
    assert isinstance(mems, list)
    assert len(mems) == 0
