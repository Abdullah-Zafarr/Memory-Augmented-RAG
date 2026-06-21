import pytest
import rag_pipeline

def test_parse_empty_pdf():
    text = rag_pipeline.parse_pdf(b"")
    assert text == ""
