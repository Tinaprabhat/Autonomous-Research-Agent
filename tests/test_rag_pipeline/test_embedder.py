"""
Unit tests for backend/app/rag_pipeline/embedder.py
"""
import numpy as np
import pytest

from backend.app.rag_pipeline.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


# ── embed_documents ───────────────────────────────────────────────────────────

def test_embed_documents_returns_array(embedder, sample_texts):
    result = embedder.embed_documents(sample_texts)
    assert hasattr(result, "__len__")


def test_embed_documents_shape(embedder, sample_texts):
    result = embedder.embed_documents(sample_texts)
    arr = np.array(result)
    assert arr.shape == (len(sample_texts), 384)


def test_embed_documents_dtype_float(embedder, sample_texts):
    result = embedder.embed_documents(sample_texts)
    arr = np.array(result)
    assert np.issubdtype(arr.dtype, np.floating)


def test_embed_documents_single_text(embedder):
    result = embedder.embed_documents(["single sentence test"])
    arr = np.array(result)
    assert arr.shape == (1, 384)


def test_embed_documents_deterministic(embedder, sample_texts):
    a = np.array(embedder.embed_documents(sample_texts[:3]))
    b = np.array(embedder.embed_documents(sample_texts[:3]))
    np.testing.assert_array_almost_equal(a, b)


# ── embed_query ───────────────────────────────────────────────────────────────

def test_embed_query_returns_1d_vector(embedder, sample_query):
    result = embedder.embed_query(sample_query)
    arr = np.array(result)
    assert arr.ndim == 1
    assert arr.shape[0] == 384


def test_embed_query_is_float(embedder, sample_query):
    result = embedder.embed_query(sample_query)
    assert np.issubdtype(np.array(result).dtype, np.floating)


def test_embed_query_deterministic(embedder, sample_query):
    a = embedder.embed_query(sample_query)
    b = embedder.embed_query(sample_query)
    np.testing.assert_array_almost_equal(a, b)


def test_embed_query_different_texts_give_different_vectors(embedder):
    v1 = embedder.embed_query("attention mechanism in transformers")
    v2 = embedder.embed_query("gradient descent optimisation")
    assert not np.allclose(v1, v2)


# ── Semantic sanity ───────────────────────────────────────────────────────────

def test_similar_sentences_closer_than_unrelated(embedder):
    v_attn1 = embedder.embed_query("self-attention in neural networks")
    v_attn2 = embedder.embed_query("transformer attention heads")
    v_unrela = embedder.embed_query("cooking pasta recipes")

    sim_related   = float(np.dot(v_attn1, v_attn2))
    sim_unrelated = float(np.dot(v_attn1, v_unrela))
    assert sim_related > sim_unrelated
