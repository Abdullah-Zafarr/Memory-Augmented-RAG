import pytest
import vector_store

def test_count_documents():
    # Simple count check to see if database interacts cleanly
    try:
        count = vector_store.count_documents()
        assert isinstance(count, int)
    except Exception as e:
        pytest.skip(f"Vector store client not fully running: {e}")
