from __future__ import annotations

from academic_chatbot.retrieval.service import RetrievalIntegrityError, RetrievalService


def test_retrieval_result_models_are_importable() -> None:
    """Would fail until Task 5 supplies a public fail-closed retrieval boundary."""
    assert issubclass(RetrievalIntegrityError, RuntimeError)
    assert RetrievalService.__name__ == "RetrievalService"
