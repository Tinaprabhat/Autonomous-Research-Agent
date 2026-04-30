"""
Unit tests for backend/app/rag_pipeline/retriever.py (basic vector search)
"""
import numpy as np
import pytest

import backend.app.rag_pipeline.vector_store as vs_module
from backend.app.rag_pipeline.vector_store import VectorStore
from backend.app.rag_pipeline.embedder import Embedder
from backend.app.rag_pipeline.retriever import Retriever


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture
def populated_store(tmp_vector_store_dir, monkeypatch, sample_chunks, embedder):
    monkeypatch.setattr(vs_module, "VECTOR_PATH", str(tmp_vector_store_dir))
    monkeypatch.setattr(vs_module, "INDEX_FILE", str(tmp_vector_store_dir / "faiss_index.bin"))
    monkeypatch.setattr(vs_module, "META_FILE",  str(tmp_vector_store_dir / "chunk_metadata.json"))

    store = VectorStore(dim=384)
    texts = [c["text"] for c in sample_chunks]
    embs = embedder.embed_documents(texts).astype("float32")
    store.add_embeddings(embs, sample_chunks)
    return store


@pytest.fixture
def retriever(populated_store, embedder):
    return Retriever(populated_store, embedder)


# ── Basic search ──────────────────────────────────────────────────────────────

def test_search_returns_list(retriever, sample_query):
    results = retriever.search(sample_query, k=3)
    assert isinstance(results, list)


def test_search_returns_at_most_k(retriever, sample_query):
    k = 3
    results = retriever.search(sample_query, k=k)
    assert len(results) <= k


def test_search_returns_dicts_with_text(retriever, sample_query):
    results = retriever.search(sample_query, k=5)
    for r in results:
        assert isinstance(r, dict)
        assert "text" in r


def test_search_results_are_from_corpus(retriever, sample_query, sample_chunks):
    results = retriever.search(sample_query, k=5)
    corpus_texts = {c["text"] for c in sample_chunks}
    for r in results:
        assert r["text"] in corpus_texts


# ── Relevance sanity ──────────────────────────────────────────────────────────

def test_attention_query_retrieves_attention_chunks(retriever):
    results = retriever.search("self-attention mechanism transformers", k=3)
    texts = " ".join(r["text"].lower() for r in results)
    assert "attention" in texts or "transformer" in texts


def test_different_queries_give_different_top_results(retriever):
    r1 = retriever.search("attention transformer", k=1)
    r2 = retriever.search("FAISS vector similarity search", k=1)
    # Top-1 results should differ for very different queries
    assert r1[0]["text"] != r2[0]["text"]


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_k_larger_than_corpus_returns_all(retriever, sample_chunks):
    results = retriever.search("anything", k=1000)
    assert len(results) <= len(sample_chunks)
