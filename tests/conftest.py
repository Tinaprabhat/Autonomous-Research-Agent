"""
Shared pytest fixtures for all test modules.
"""
import numpy as np
import pytest


# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
    "Transformers use self-attention to compute representations of sequences.",
    "BERT pre-trains deep bidirectional transformers for language understanding.",
    "RAG combines retrieval of documents with generative language models.",
    "Vector databases store high-dimensional embeddings for fast similarity search.",
    "BM25 is a sparse keyword-based ranking function used in information retrieval.",
    "Cross-encoders score query-document pairs jointly for accurate reranking.",
    "Fine-tuning adapts pre-trained models to specific downstream tasks.",
    "LoRA reduces trainable parameters by injecting low-rank decomposition matrices.",
    "FAISS enables efficient similarity search over millions of high-dimensional vectors.",
]

SAMPLE_CHUNKS = [
    {"text": t, "chunk_id": i, "source": f"paper_{i // 3}.pdf"}
    for i, t in enumerate(SAMPLE_TEXTS)
]

SAMPLE_QUERY = "How does attention work in transformers?"

SAMPLE_DOCUMENT_TEXT = """
Abstract

This paper introduces a novel approach to natural language processing using transformer
architectures. The self-attention mechanism allows models to capture long-range dependencies
without recurrence, making training highly parallelizable.

Introduction

Traditional sequence models relied on recurrent neural networks (RNNs) to process sequences.
However, RNNs struggle with very long sequences due to vanishing gradients and sequential
computation that prevents parallelization during training.

The transformer model replaces recurrence entirely with attention mechanisms. Given queries Q,
keys K, and values V, the attention is computed as: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.

Methodology

We train our model on a large corpus of text using masked language modelling. The architecture
consists of 12 transformer blocks with 768 hidden dimensions and 12 attention heads. We use
the Adam optimizer with a warm-up learning rate schedule.

Results

Our model achieves state-of-the-art performance on multiple benchmarks including GLUE and
SuperGLUE. The approach outperforms previous methods by a significant margin while requiring
less compute than ensemble approaches.

Conclusion

We presented an efficient transformer-based approach to NLP. Future work will explore scaling
to larger model sizes and extending the pre-training corpus to cover more languages.
"""


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_texts():
    return SAMPLE_TEXTS


@pytest.fixture
def sample_chunks():
    return [dict(c) for c in SAMPLE_CHUNKS]  # fresh copy each test


@pytest.fixture
def sample_query():
    return SAMPLE_QUERY


@pytest.fixture
def sample_document_text():
    return SAMPLE_DOCUMENT_TEXT


@pytest.fixture
def sample_embeddings():
    """384-dim random unit vectors matching all-MiniLM-L6-v2 output shape."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((len(SAMPLE_TEXTS), 384)).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def tmp_vector_store_dir(tmp_path):
    """Temporary directory for FAISS index and metadata."""
    d = tmp_path / "vector_store"
    d.mkdir()
    return d
