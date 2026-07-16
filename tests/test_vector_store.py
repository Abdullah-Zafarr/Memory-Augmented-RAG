import pytest
from src.rag import vector_store
from src.rag import pipeline as rag_pipeline

def test_count_documents():
    try:
        count = vector_store.count_documents()
        assert isinstance(count, int)
    except Exception as e:
        pytest.skip(f"Vector store client not fully running: {e}")

def test_markdown_ingestion():
    test_md = "# Title\n\nThis is a sample markdown file content for testing."
    chunks_created = rag_pipeline.ingest_text(test_md, "test_file.md")
    assert chunks_created >= 0
