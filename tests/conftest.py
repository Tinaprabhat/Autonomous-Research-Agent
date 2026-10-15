"""
Shared pytest fixtures for all test modules.
"""
import os

# Never let a test talk to a real (cloud) Qdrant, whatever .env says.
os.environ["QDRANT_FORCE_LOCAL"] = "1"

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


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder (384-d), no model download."""

    def _vec(self, text):
        import re, zlib
        v = np.zeros(384, dtype="float32")
        for tok in re.findall(r"\w+", text.lower()):
            v[zlib.crc32(tok.encode()) % 384] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts):
        return np.array([self._vec(t) for t in texts])

    def embed_query(self, q):
        return self._vec(q)


class FakeVerifier:
    """Entailment-chain stand-in: a claim is supported iff every claim word of length>3 appears in the cited chunks."""

    name = "fake"

    def verify(self, items):
        import re
        flags = []
        for texts, claim in items:
            words = [w for w in re.findall(r"\w+", claim.lower()) if len(w) > 3]
            blob = " ".join(texts).lower()
            flags.append(bool(words) and all(w in blob for w in words))
        return flags, "fake"


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_verifier():
    return FakeVerifier()


@pytest.fixture
def mem_store(fake_embedder):
    """In-memory Qdrant store loaded with provenance-rich sample chunks."""
    from backend.app.storage.qdrant_store import QdrantStore, make_client
    chunks = []
    for i, t in enumerate(SAMPLE_TEXTS):
        chunks.append({
            "text": t, "chunk_id": f"2401.0000{i // 3}_v1_{i}",
            "arxiv_id": f"2401.0000{i // 3}", "version": "v1",
            "title": f"Paper {i // 3}", "author": ["Ada Lovelace"] if i % 2 else ["Alan Turing"],
            "category": "cs.CL" if i < 5 else "cs.LG",
            "published_date": f"2024-0{(i % 3) + 1}-15T00:00:00Z",
            "section": "Method" if i % 2 else "Introduction", "page": i + 1,
            "char_start": 0, "char_end": len(t),
        })
    store = QdrantStore(make_client(path=":memory:"), collection="test")
    store.add_embeddings(fake_embedder.embed_documents([c["text"] for c in chunks]), chunks)
    return store
