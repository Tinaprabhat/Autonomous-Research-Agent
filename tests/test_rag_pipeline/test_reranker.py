"""
Unit tests for backend/app/rag_pipeline/reranker.py
"""
import pytest

from backend.app.rag_pipeline.reranker import Reranker


@pytest.fixture(scope="module")
def reranker():
    return Reranker()


# ── Basic output shape ────────────────────────────────────────────────────────

def test_rerank_returns_list(reranker, sample_chunks, sample_query):
    result = reranker.rerank(sample_query, sample_chunks, top_k=5)
    assert isinstance(result, list)


def test_rerank_returns_at_most_top_k(reranker, sample_chunks, sample_query):
    result = reranker.rerank(sample_query, sample_chunks, top_k=3)
    assert len(result) <= 3


def test_rerank_preserves_dict_type(reranker, sample_chunks, sample_query):
    result = reranker.rerank(sample_query, sample_chunks, top_k=5)
    for r in result:
        assert isinstance(r, dict)
        assert "text" in r


def test_rerank_works_with_string_list(reranker, sample_texts, sample_query):
    result = reranker.rerank(sample_query, sample_texts, top_k=3)
    assert isinstance(result, list)
    assert all(isinstance(r, str) for r in result)


# ── Empty input ───────────────────────────────────────────────────────────────

def test_rerank_empty_documents_returns_empty(reranker, sample_query):
    assert reranker.rerank(sample_query, [], top_k=5) == []


def test_rerank_top_k_zero_returns_empty(reranker, sample_chunks, sample_query):
    result = reranker.rerank(sample_query, sample_chunks, top_k=0)
    assert result == []


# ── Ordering sanity ───────────────────────────────────────────────────────────

def test_most_relevant_chunk_ranked_first(reranker):
    query = "attention mechanism in transformers"
    docs = [
        {"text": "Cooking pasta requires boiling water.", "chunk_id": 0},
        {"text": "Self-attention lets transformers relate tokens across the sequence.", "chunk_id": 1},
        {"text": "Football match results from last weekend.", "chunk_id": 2},
    ]
    result = reranker.rerank(query, docs, top_k=3)
    assert result[0]["chunk_id"] == 1, "Attention-related chunk should rank first"


def test_rerank_top_k_larger_than_docs_returns_all(reranker, sample_chunks, sample_query):
    result = reranker.rerank(sample_query, sample_chunks[:3], top_k=10)
    assert len(result) == 3


# ── Idempotency ───────────────────────────────────────────────────────────────

def test_rerank_same_query_is_deterministic(reranker, sample_chunks, sample_query):
    r1 = reranker.rerank(sample_query, sample_chunks, top_k=3)
    r2 = reranker.rerank(sample_query, sample_chunks, top_k=3)
    assert [d["chunk_id"] for d in r1] == [d["chunk_id"] for d in r2]
