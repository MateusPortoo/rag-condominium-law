"""Verify constants are centralized — single source of truth in core/config.

These tests guard against the COLLECTION_NAME bug found in the audit:
indexer.py indexed to "condominio_docs" while dense.py searched "rag_condominios",
causing zero results silently. All modules must import the same value.
"""

from rag_condominios.core.config import (
    COLLECTION_NAME,
    CRAG_AMBIGUOUS_THRESHOLD,
    CRAG_CORRECT_THRESHOLD,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RRF_K,
    TIKTOKEN_ENCODING,
    TOP_K_DEFAULT,
)
from rag_condominios.ingest.indexer import COLLECTION_NAME as INDEXER_CN
from rag_condominios.retrieval.dense import COLLECTION_NAME as DENSE_CN
from rag_condominios.retrieval.dense import EMBEDDING_MODEL as DENSE_EM
from rag_condominios.retrieval.dense import TOP_K_DEFAULT as DENSE_TOP_K
from rag_condominios.retrieval.fusion import RRF_K as FUSION_RRF_K
from rag_condominios.retrieval.fusion import TOP_K_DEFAULT as FUSION_TOP_K
from rag_condominios.retrieval.sparse import TOP_K_DEFAULT as SPARSE_TOP_K


def test_collection_name_identical_in_indexer_and_dense() -> None:
    assert INDEXER_CN == DENSE_CN, (
        f"COLLECTION_NAME mismatch: indexer uses '{INDEXER_CN}', dense uses '{DENSE_CN}'. "
        "The pipeline indexes to one collection and searches another — zero results in production."
    )


def test_collection_name_matches_config() -> None:
    assert INDEXER_CN == COLLECTION_NAME
    assert DENSE_CN == COLLECTION_NAME


def test_embedding_model_consistent() -> None:
    assert DENSE_EM == EMBEDDING_MODEL


def test_top_k_consistent_across_modules() -> None:
    assert DENSE_TOP_K == TOP_K_DEFAULT
    assert FUSION_TOP_K == TOP_K_DEFAULT
    assert SPARSE_TOP_K == TOP_K_DEFAULT


def test_rrf_k_consistent() -> None:
    assert FUSION_RRF_K == RRF_K


def test_tiktoken_encoding_defined() -> None:
    assert TIKTOKEN_ENCODING == "cl100k_base"


def test_collection_name_value() -> None:
    assert COLLECTION_NAME == "condominio_docs"


def test_crag_thresholds_defined() -> None:
    assert CRAG_CORRECT_THRESHOLD == 0.75
    assert CRAG_AMBIGUOUS_THRESHOLD == 0.70


def test_correct_threshold_greater_than_ambiguous() -> None:
    assert CRAG_CORRECT_THRESHOLD > CRAG_AMBIGUOUS_THRESHOLD


def test_reranker_model_defined() -> None:
    assert RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
