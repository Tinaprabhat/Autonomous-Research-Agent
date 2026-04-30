"""
Unit tests for backend/app/rag_pipeline/hybrid_retriever.py
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from backend.app.rag_pipeline.hybrid_retriever import HybridRetriever


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_store(sample_chunks, sample_embeddings):
    """Minimal fake VectorStore whose .search() returns first k chunks."""
    store = MagicMock()
    store.search.side_effect = lambda query, k: sample_chunks[:k]
    return store


@pytest.fixture
def hybrid(sample_chunks, sample_embeddings):
    store = _make_store(sample_chunks, sample_embeddings)
    return HybridRetriever(store, sample_chunks)


# ── Basic correctness ─────────────────────────────────────────────────────────

def test_search_returns_list(hybrid, sample_query):
    results = hybrid.search(sample_query, k=5)
    assert isinstance(results, list)


def test_search_returns_at_most_k(hybrid, sample_query):
    k = 4
    results = hybrid.search(sample_query, k=k)
    assert len(results) <= k


def test_search_results_have_text_key(hybrid, sample_query):
    results = hybrid.search(sample_query, k=5)
    for r in results:
        assert "text" in r


def test_search_results_have_chunk_id(hybrid, sample_query):
    results = hybrid.search(sample_query, k=5)
    for r in results:
        assert "chunk_id" in r


# ── Deduplication ─────────────────────────────────────────────────────────────

def test_no_duplicate_chunk_ids(hybrid, sample_query):
    results = hybrid.search(sample_query, k=10)
    ids = [r["chunk_id"] for r in results]
    assert len(ids) == len(set(ids))


# ── BM25 contribution ─────────────────────────────────────────────────────────

def test_keyword_match_appears_in_results(sample_chunks):
    """A query that exactly matches one chunk's text should retrieve it."""
    target = sample_chunks[5]  # "BM25 is a sparse keyword-based ranking function..."
    store = MagicMock()
    store.search.return_value = []  # vector search returns nothing

    hybrid = HybridRetriever(store, sample_chunks)
    results = hybrid.search("BM25 keyword ranking", k=5)

    texts = [r["text"] for r in results]
    assert target["text"] in texts


# ── Vector-only fallback ──────────────────────────────────────────────────────

def test_vector_results_included_when_bm25_weak(sample_chunks):
    store = MagicMock()
    store.search.return_value = sample_chunks[:3]

    hybrid = HybridRetriever(store, sample_chunks)
    results = hybrid.search("xyzzy nonexistent token qqqq", k=5)

    # Vector results should still be returned
    assert len(results) >= 1


# ── Empty corpus ─────────────────────────────────────────────────────────────

def test_empty_documents_returns_empty():
    store = MagicMock()
    store.search.return_value = []
    hybrid = HybridRetriever(store, [])
    results = hybrid.search("anything", k=5)
    assert results == []
