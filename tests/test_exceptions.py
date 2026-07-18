from src.core.exceptions import DatabaseConnectionError, RAGException
import pytest

def test_exception_inheritance():
    exc = DatabaseConnectionError("Test connection failure")
    assert isinstance(exc, RAGException)
    assert str(exc) == "Test connection failure"
