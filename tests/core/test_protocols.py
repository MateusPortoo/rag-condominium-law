"""Verify Protocol definitions and conformance."""

from unittest.mock import MagicMock

from rag_condominios.core.protocols import BM25Retriever, LLMClient


def test_bm25_retriever_is_runtime_checkable() -> None:
    mock = MagicMock()
    mock.retrieve = MagicMock(return_value=([], []))
    assert isinstance(mock, BM25Retriever)


def test_object_without_retrieve_does_not_satisfy_bm25_retriever() -> None:
    class NotARetriever:
        pass

    assert not isinstance(NotARetriever(), BM25Retriever)


def test_llm_client_is_runtime_checkable() -> None:
    mock = MagicMock()
    type(mock).chat = MagicMock()
    assert isinstance(mock, LLMClient)


def test_bm25_retriever_protocol_exposes_retrieve() -> None:
    assert hasattr(BM25Retriever, "retrieve")


def test_llm_client_protocol_exposes_chat() -> None:
    assert hasattr(LLMClient, "chat")
