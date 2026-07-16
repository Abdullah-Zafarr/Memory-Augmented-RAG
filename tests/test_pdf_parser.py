import pytest
from src.rag import pipeline as rag_pipeline

def test_parse_empty_pdf():
    text = rag_pipeline.parse_pdf(b"")
    assert text == ""
