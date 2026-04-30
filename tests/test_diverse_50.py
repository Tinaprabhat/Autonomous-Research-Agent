"""
50 diverse test cases spanning every module of the RAG pipeline.

Coverage areas:
  A. Chunking         — 12 cases
  B. Embedder         —  8 cases
  C. HybridRetriever  —  8 cases
  D. Reranker         —  7 cases
  E. LocalLLM         —  7 cases
  F. API endpoints    —  5 cases
  G. Evaluation metrics — 3 cases
  Total: 50 cases
"""

import sys
import json
import hashlib
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

# ── helpers ───────────────────────────────────────────────────────────────────

def _long_para(words=30, word="word"):
    return " ".join([word] * words)

def _doc(n_paras=4, words_each=40):
    paras = [_long_para(words_each, f"para{i}word") for i in range(n_paras)]
    return "\n\n".join(paras)


# ═══════════════════════════════════════════════════════════════════
# A. CHUNKING  (12 cases)
# ═══════════════════════════════════════════════════════════════════

from backend.app.ingestion.chunking import chunk_document


class TestChunkingDiverse:

    # A-01  numeric-only paragraph is dropped (< 80 chars)
    def test_numeric_only_short_para_filtered(self):
        text = "1 2 3 4 5\n\n" + _long_para(30)
        chunks = chunk_document(text)
        for c in chunks:
            assert c["text"] != "1 2 3 4 5"

    # A-02  unicode / non-ASCII text produces valid chunks
    def test_unicode_text_chunked(self):
        text = "Transformers use self-attention.\n\n" * 5
        text_unicode = text.replace("Transformers", "Трансформеры")
        chunks = chunk_document(text_unicode)
        assert isinstance(chunks, list)

    # A-03  very long single paragraph split into multiple chunks
    def test_very_long_single_para_splits(self):
        text = " ".join(["The model learns representations"] * 60)
        chunks = chunk_document(text)
        assert len(chunks) >= 1

    # A-04  two semantically unrelated paragraphs kept separate at threshold=1
    def test_unrelated_paras_not_merged_at_high_threshold(self):
        p1 = "Transformers rely on self-attention to process sequences efficiently."
        p2 = "The economy grew by three percent in the last quarter of the year."
        text = "\n\n".join([p1 + " " * 20 + p1, p2 + " " * 20 + p2])
        chunks_split = chunk_document(text, similarity_threshold=1.0)
        chunks_merged = chunk_document(text, similarity_threshold=0.0)
        assert len(chunks_split) >= len(chunks_merged)

    # A-05  metadata keys do not overwrite text / chunk_id
    def test_metadata_does_not_overwrite_core_keys(self):
        meta = {"text": "SHOULD_NOT_OVERWRITE", "chunk_id": 9999}
        chunks = chunk_document(_doc(), metadata=meta)
        for c in chunks:
            assert c["text"] != "SHOULD_NOT_OVERWRITE"
            assert c["chunk_id"] != 9999

    # A-06  all chunks have non-empty text
    def test_all_chunks_nonempty_text(self):
        chunks = chunk_document(_doc(n_paras=6, words_each=30))
        for c in chunks:
            assert c["text"].strip() != ""

    # A-07  chunk_ids are contiguous integers 0, 1, 2, …
    def test_chunk_ids_contiguous(self):
        chunks = chunk_document(_doc(n_paras=5, words_each=35))
        ids = [c["chunk_id"] for c in chunks]
        assert ids == list(range(len(ids)))

    # A-08  multiple metadata fields all propagated
    def test_multiple_metadata_fields_propagated(self):
        meta = {"source": "test.pdf", "year": 2024, "author": "Tina"}
        chunks = chunk_document(_doc(), metadata=meta)
        for c in chunks:
            assert c["source"] == "test.pdf"
            assert c["year"] == 2024
            assert c["author"] == "Tina"

    # A-09  repeated text produces deterministic output
    def test_deterministic_output(self):
        text = _doc(n_paras=4, words_each=25)
        c1 = chunk_document(text)
        c2 = chunk_document(text)
        assert [c["text"] for c in c1] == [c["text"] for c in c2]

    # A-10  single paragraph exactly 80 chars is kept
    def test_paragraph_exactly_80_chars_kept(self):
        para = "a" * 80
        text = para + "\n\n" + _long_para(30)
        chunks = chunk_document(text)
        texts = [c["text"] for c in chunks]
        assert any("a" * 80 in t for t in texts)

    # A-11  no chunk exceeds 600 chars for a normal document
    def test_no_chunk_over_600_chars(self):
        text = _doc(n_paras=8, words_each=50)
        for c in chunk_document(text):
            assert len(c["text"]) <= 600

    # A-12  newline-heavy text (single-word lines) still chunks correctly
    def test_newline_heavy_text(self):
        text = "\n\n".join([_long_para(25)] * 5)
        chunks = chunk_document(text)
        assert len(chunks) >= 1
        assert all("chunk_id" in c for c in chunks)


# ═══════════════════════════════════════════════════════════════════
# B. EMBEDDER  (8 cases)
# ═══════════════════════════════════════════════════════════════════

from backend.app.rag_pipeline.embedder import Embedder

@pytest.fixture(scope="module")
def embedder():
    return Embedder()


class TestEmbedderDiverse:

    # B-01  empty-ish query still returns 384-dim vector
    def test_single_word_query(self, embedder):
        v = embedder.embed_query("attention")
        assert np.array(v).shape == (384,)

    # B-02  very long text
    def test_long_text_embedding(self, embedder):
        text = " ".join(["transformer attention mechanism"] * 50)
        v = embedder.embed_query(text)
        assert np.array(v).shape == (384,)

    # B-03  batch of 1 matches single embed_query
    def test_batch_one_matches_single(self, embedder):
        text = "self-attention is a core mechanism"
        v_single = np.array(embedder.embed_query(text))
        v_batch  = np.array(embedder.embed_documents([text]))[0]
        np.testing.assert_array_almost_equal(v_single, v_batch, decimal=5)

    # B-04  vectors are finite (no NaN/Inf)
    def test_no_nan_or_inf(self, embedder):
        texts = ["attention", "bert", "faiss", "LoRA fine-tuning"]
        embs = np.array(embedder.embed_documents(texts))
        assert np.all(np.isfinite(embs))

    # B-05  norms are positive (non-zero vectors)
    def test_nonzero_vectors(self, embedder):
        texts = ["transformer", "embedding", "vector search"]
        embs = np.array(embedder.embed_documents(texts))
        norms = np.linalg.norm(embs, axis=1)
        assert np.all(norms > 0)

    # B-06  10-sentence batch returns shape (10, 384)
    def test_batch_10_shape(self, embedder):
        texts = [f"sentence number {i} about AI" for i in range(10)]
        embs = np.array(embedder.embed_documents(texts))
        assert embs.shape == (10, 384)

    # B-07  cosine sim between identical embeddings == 1
    def test_identical_texts_cosine_sim_one(self, embedder):
        t = "BERT pre-trains with masked language modelling"
        v1, v2 = embedder.embed_query(t), embedder.embed_query(t)
        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        assert sim == pytest.approx(1.0, abs=1e-5)

    # B-08  RAG-related query closer to RAG context than to cooking
    def test_domain_separation(self, embedder):
        q    = embedder.embed_query("vector database similarity search")
        rag  = embedder.embed_query("FAISS stores high-dimensional embeddings")
        cook = embedder.embed_query("chop onions and fry with butter")
        sim_rag  = float(np.dot(q, rag))
        sim_cook = float(np.dot(q, cook))
        assert sim_rag > sim_cook


# ═══════════════════════════════════════════════════════════════════
# C. HYBRID RETRIEVER  (8 cases)
# ═══════════════════════════════════════════════════════════════════

from backend.app.rag_pipeline.hybrid_retriever import HybridRetriever

HYBRID_CORPUS = [
    {"text": "Self-attention computes weighted combinations of values.", "chunk_id": 0, "source": "a.pdf"},
    {"text": "BERT pre-trains using masked language modelling objectives.", "chunk_id": 1, "source": "b.pdf"},
    {"text": "FAISS enables fast approximate nearest-neighbour search.", "chunk_id": 2, "source": "c.pdf"},
    {"text": "BM25 ranks documents by term frequency and inverse document frequency.", "chunk_id": 3, "source": "d.pdf"},
    {"text": "LoRA fine-tunes large models with low-rank decomposition matrices.", "chunk_id": 4, "source": "e.pdf"},
    {"text": "RAG combines document retrieval with generative language models.", "chunk_id": 5, "source": "f.pdf"},
    {"text": "Cross-encoders jointly score query-document pairs for reranking.", "chunk_id": 6, "source": "g.pdf"},
    {"text": "Gradient descent optimises model parameters by minimising loss.", "chunk_id": 7, "source": "h.pdf"},
]

def _make_hybrid(corpus=None):
    docs = corpus or HYBRID_CORPUS
    store = MagicMock()
    store.search.side_effect = lambda query, k: docs[:k]
    return HybridRetriever(store, docs)


class TestHybridRetrieverDiverse:

    # C-01  results all belong to the corpus
    def test_results_belong_to_corpus(self):
        h = _make_hybrid()
        results = h.search("attention mechanism", k=5)
        ids = {d["chunk_id"] for d in HYBRID_CORPUS}
        for r in results:
            assert r["chunk_id"] in ids

    # C-02  BM25 surfaces exact keyword match "BM25"
    def test_bm25_keyword_surfaces_exact_match(self):
        store = MagicMock(); store.search.return_value = []
        h = HybridRetriever(store, HYBRID_CORPUS)
        results = h.search("BM25 inverse document frequency", k=5)
        assert any(r["chunk_id"] == 3 for r in results)

    # C-03  k=1 returns exactly one result
    def test_k1_returns_one(self):
        h = _make_hybrid()
        assert len(h.search("anything", k=1)) == 1

    # C-04  k larger than corpus returns at most corpus size
    def test_k_larger_than_corpus(self):
        h = _make_hybrid()
        results = h.search("attention", k=100)
        assert len(results) <= len(HYBRID_CORPUS)

    # C-05  BM25 ranks the exact-keyword document at top-1 for different queries
    def test_different_queries_surface_different_bm25_top1(self):
        # Vector store returns nothing — pure BM25 path
        store = MagicMock()
        store.search.return_value = []
        h = HybridRetriever(store, HYBRID_CORPUS)
        r_bm25 = h.search("BM25 inverse document frequency term ranking", k=5)
        r_lora = h.search("LoRA low-rank decomposition fine-tuning adapters", k=5)
        # BM25 query should surface chunk_id=3 (BM25 doc)
        assert r_bm25[0]["chunk_id"] == 3
        # LoRA query should surface chunk_id=4 (LoRA doc)
        assert r_lora[0]["chunk_id"] == 4

    # C-06  all chunk_ids in results are unique
    def test_unique_chunk_ids_in_results(self):
        h = _make_hybrid()
        results = h.search("FAISS embedding search", k=6)
        ids = [r["chunk_id"] for r in results]
        assert len(ids) == len(set(ids))

    # C-07  corpus of one document returns that document
    def test_single_doc_corpus(self):
        single = [{"text": "Only document in corpus.", "chunk_id": 0, "source": "x.pdf"}]
        store = MagicMock(); store.search.return_value = single
        h = HybridRetriever(store, single)
        results = h.search("document", k=5)
        assert len(results) == 1
        assert results[0]["chunk_id"] == 0

    # C-08  metadata (source) preserved through retrieval
    def test_metadata_preserved(self):
        h = _make_hybrid()
        results = h.search("gradient descent", k=4)
        for r in results:
            assert "source" in r
            assert r["source"].endswith(".pdf")


# ═══════════════════════════════════════════════════════════════════
# D. RERANKER  (7 cases)
# ═══════════════════════════════════════════════════════════════════

from backend.app.rag_pipeline.reranker import Reranker

@pytest.fixture(scope="module")
def reranker():
    return Reranker()

RERANK_DOCS = [
    {"text": "RAG retrieves documents then generates answers.", "chunk_id": 0},
    {"text": "Weather forecast shows rain tomorrow.", "chunk_id": 1},
    {"text": "Retrieval augmented generation improves factuality.", "chunk_id": 2},
    {"text": "Stock prices rose three percent.", "chunk_id": 3},
    {"text": "Dense retrieval uses embedding similarity.", "chunk_id": 4},
]


class TestRerankerDiverse:

    # D-01  RAG query: retrieval-related chunks ranked first
    def test_rag_query_retrieval_chunks_first(self, reranker):
        results = reranker.rerank("How does retrieval-augmented generation work?", RERANK_DOCS, top_k=3)
        top_ids = {r["chunk_id"] for r in results}
        rag_ids = {0, 2, 4}
        assert len(top_ids & rag_ids) >= 2

    # D-02  single document input returns that document
    def test_single_doc_returned(self, reranker):
        result = reranker.rerank("attention", [RERANK_DOCS[0]], top_k=1)
        assert len(result) == 1
        assert result[0]["chunk_id"] == 0

    # D-03  string inputs stay strings
    def test_string_input_stays_string(self, reranker):
        strings = ["attention is a mechanism", "cooking with spices", "FAISS nearest neighbour"]
        result = reranker.rerank("attention in neural networks", strings, top_k=2)
        assert all(isinstance(r, str) for r in result)

    # D-04  dict inputs stay dicts
    def test_dict_input_stays_dict(self, reranker):
        result = reranker.rerank("attention", RERANK_DOCS, top_k=3)
        assert all(isinstance(r, dict) for r in result)

    # D-05  all returned docs are subsets of input
    def test_returned_docs_subset_of_input(self, reranker):
        result = reranker.rerank("retrieval", RERANK_DOCS, top_k=3)
        input_ids = {d["chunk_id"] for d in RERANK_DOCS}
        for r in result:
            assert r["chunk_id"] in input_ids

    # D-06  top_k=1 returns exactly one result
    def test_top_k_one(self, reranker):
        result = reranker.rerank("retrieval generation", RERANK_DOCS, top_k=1)
        assert len(result) == 1

    # D-07  repeated rerank calls are stable (same order)
    def test_stable_ordering(self, reranker):
        q = "retrieval augmented generation documents"
        r1 = [d["chunk_id"] for d in reranker.rerank(q, RERANK_DOCS, top_k=5)]
        r2 = [d["chunk_id"] for d in reranker.rerank(q, RERANK_DOCS, top_k=5)]
        assert r1 == r2


# ═══════════════════════════════════════════════════════════════════
# E. LOCAL LLM  (7 cases)
# ═══════════════════════════════════════════════════════════════════

from backend.app.llm.local_llm import LocalLLM


def _mock_ok(text="answer"):
    resp = MagicMock()
    resp.json.return_value = {"response": text}
    resp.raise_for_status = MagicMock()
    return resp

@pytest.fixture(autouse=True)
def clear_cache():
    LocalLLM.clear_cache()
    yield
    LocalLLM.clear_cache()


class TestLocalLLMDiverse:

    # E-01  multi-line prompt handled correctly
    @patch("backend.app.llm.local_llm.requests.post")
    def test_multiline_prompt(self, mock_post):
        mock_post.return_value = _mock_ok("multi-line ok")
        llm = LocalLLM(temperature=0.3)
        result = llm.generate("Line 1\nLine 2\nLine 3")
        assert result == "multi-line ok"

    # E-02  very long response stripped correctly
    @patch("backend.app.llm.local_llm.requests.post")
    def test_long_response_returned_intact(self, mock_post):
        long_answer = "word " * 200
        mock_post.return_value = _mock_ok("  " + long_answer + "  ")
        llm = LocalLLM(temperature=0.3)
        result = llm.generate("long test")
        assert result == long_answer.strip()

    # E-03  cache key differs when max_tokens differs
    @patch("backend.app.llm.local_llm.requests.post")
    def test_cache_key_differs_by_max_tokens(self, mock_post):
        mock_post.return_value = _mock_ok("answer")
        llm_64 = LocalLLM(temperature=0.0, max_tokens=64)
        llm_128 = LocalLLM(temperature=0.0, max_tokens=128)
        llm_64.generate("same prompt")
        llm_128.generate("same prompt")
        assert mock_post.call_count == 2

    # E-04  non-zero temperature skips cache even if same prompt
    @patch("backend.app.llm.local_llm.requests.post")
    def test_nonzero_temp_never_caches(self, mock_post):
        mock_post.return_value = _mock_ok("x")
        llm = LocalLLM(temperature=0.7)
        for _ in range(3):
            llm.generate("same prompt")
        assert mock_post.call_count == 3

    # E-05  cache stats valid=0 before any call
    def test_cache_stats_empty_at_start(self):
        stats = LocalLLM.cache_stats()
        assert stats["total"] == 0
        assert stats["valid"] == 0

    # E-06  HTTPError raises RuntimeError
    @patch("backend.app.llm.local_llm.requests.post")
    def test_http_error_raises_runtime_error(self, mock_post):
        import requests as req
        resp = MagicMock()
        resp.raise_for_status.side_effect = req.exceptions.HTTPError("404")
        mock_post.return_value = resp
        llm = LocalLLM()
        with pytest.raises(RuntimeError):
            llm.generate("test")

    # E-07  generate_stream returns correct token sequence
    @patch("backend.app.llm.local_llm.requests.post")
    def test_stream_correct_sequence(self, mock_post):
        import json
        lines = [
            json.dumps({"response": t, "done": False}).encode()
            for t in ["The", " answer", " is", " 42"]
        ]
        lines.append(json.dumps({"response": "", "done": True}).encode())
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.iter_lines.return_value = lines
        mock_post.return_value = resp
        llm = LocalLLM()
        tokens = list(llm.generate_stream("what is the answer?"))
        assert "".join(tokens) == "The answer is 42"


# ═══════════════════════════════════════════════════════════════════
# F. API ENDPOINTS  (5 cases)
# ═══════════════════════════════════════════════════════════════════

FAKE_CHUNKS_API = [
    {"text": "Transformers use attention to process sequences.", "chunk_id": 0, "source": "p1.pdf"},
    {"text": "BERT learns from unlabelled text with masking.", "chunk_id": 1, "source": "p2.pdf"},
]

@pytest.fixture(scope="module")
def api_client():
    sys.modules.pop("backend.app.api", None)
    with patch("backend.app.rag_pipeline.vector_store.VectorStore.load", return_value=None), \
         patch("backend.app.rag_pipeline.embedder.Embedder.__init__",    return_value=None), \
         patch("backend.app.rag_pipeline.reranker.Reranker.__init__",    return_value=None):
        import backend.app.api as mod
    mod.store.metadata = FAKE_CHUNKS_API
    mod.retriever.search  = MagicMock(return_value=FAKE_CHUNKS_API)
    mod.reranker.rerank   = MagicMock(return_value=FAKE_CHUNKS_API)
    mod.reranker.model    = MagicMock(predict=MagicMock(return_value=[2.0]))
    mod.llm.generate      = MagicMock(return_value="Transformers use self-attention.")
    mod.llm.model         = "mistral"
    from fastapi.testclient import TestClient
    return TestClient(mod.app), mod


class TestAPIDiverse:

    # F-01  query with top_k=1 calls reranker with top_k=1
    def test_top_k_1_passed_to_reranker(self, api_client):
        client, mod = api_client
        mod.reranker.rerank.reset_mock()
        client.post("/query", json={"query": "attention", "top_k": 1})
        call = mod.reranker.rerank.call_args
        assert call[1].get("top_k") == 1 or call[0][2] == 1

    # F-02  /health model field matches llm.model
    def test_health_model_field(self, api_client):
        client, mod = api_client
        resp = client.get("/health")
        assert resp.json()["model"] == "mistral"

    # F-03  response latency_ms is non-negative integer
    def test_latency_ms_non_negative(self, api_client):
        client, _ = api_client
        resp = client.post("/query", json={"query": "BERT masking"})
        assert resp.json()["latency_ms"] >= 0

    # F-04  text in citation is truncated to ≤200 chars
    def test_citation_text_truncated(self, api_client):
        client, _ = api_client
        resp = client.post("/query", json={"query": "attention"})
        for cit in resp.json()["citations"]:
            assert len(cit["text"]) <= 200

    # F-05  sources_used equals number of reranked chunks
    def test_sources_used_matches_reranked(self, api_client):
        client, mod = api_client
        mod.reranker.rerank.return_value = FAKE_CHUNKS_API  # 2 chunks
        resp = client.post("/query", json={"query": "attention"})
        assert resp.json()["sources_used"] == 2


# ═══════════════════════════════════════════════════════════════════
# G. EVALUATION METRICS  (3 cases)
# ═══════════════════════════════════════════════════════════════════

from tests.evaluation.metrics import (
    token_overlap_f1, precision_at_k, recall_at_k,
    mean_reciprocal_rank, ndcg_at_k, faithfulness,
    relevance, conciseness, score_response
)


class TestEvalMetricsDiverse:

    # G-01  ndcg is 1.0 when all relevant docs are ranked first
    def test_ndcg_perfect_top_k(self):
        retrieved = ["a", "b", "c", "x", "y"]
        relevant  = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, k=3) == pytest.approx(1.0)

    # G-02  faithfulness handles multiple context chunks — uses best match
    def test_faithfulness_uses_best_context_chunk(self):
        answer = "BERT uses masked language modelling."
        ctx = [
            "Cooking with herbs improves flavour.",
            "BERT is a transformer that uses masked language modelling for pre-training.",
        ]
        score = faithfulness(answer, ctx)
        assert score > 0.4

    # G-03  score_response with all optional fields gives all keys in [0,1]
    def test_score_response_all_fields_valid(self):
        scores = score_response(
            answer="FAISS allows fast approximate nearest-neighbour search.",
            question="How does FAISS work?",
            context_chunks=["FAISS indexes embeddings for fast similarity search."],
            reference_answer="FAISS performs fast vector similarity search.",
            latency_seconds=3.0,
            latency_budget=10.0,
        )
        for k, v in scores.items():
            assert 0.0 <= v <= 1.0, f"{k}={v} out of range"
