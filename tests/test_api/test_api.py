"""
Integration-style tests for backend/app/api.py using FastAPI TestClient.

Only VectorStore.load() is patched (prevents FAISS file I/O).
The retriever.search, reranker.rerank, and llm.generate methods are replaced
with mocks after the module is imported so all request handling is tested.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch


FAKE_CHUNKS = [
    {"text": "Attention allows transformers to relate all positions.", "chunk_id": 0, "source": "data/raw_papers/paper_a.pdf"},
    {"text": "BERT pre-trains on masked language modelling.", "chunk_id": 1, "source": "data/raw_papers/paper_b.pdf"},
    {"text": "RAG combines retrieval with generation.", "chunk_id": 2, "source": "data/raw_papers/paper_a.pdf"},
]

FAKE_ANSWER = "Attention allows models to focus on relevant tokens."


@pytest.fixture(scope="module")
def api():
    """Return (TestClient, api_module) after patching FAISS I/O."""
    sys.modules.pop("backend.app.api", None)

    with patch("backend.app.rag_pipeline.vector_store.VectorStore.load", return_value=None), \
         patch("backend.app.rag_pipeline.embedder.Embedder.__init__",    return_value=None), \
         patch("backend.app.rag_pipeline.reranker.Reranker.__init__",    return_value=None):

        import backend.app.api as api_module

    # VectorStore.__init__ ran normally → store.metadata = [] and store.index exists.
    # Inject fake data so endpoint logic has something to work with.
    api_module.store.metadata = FAKE_CHUNKS
    api_module.retriever.search  = MagicMock(return_value=FAKE_CHUNKS)
    api_module.reranker.rerank   = MagicMock(return_value=FAKE_CHUNKS)
    # reranker.model is needed by citation_guard; predict scores above 0.0 threshold
    api_module.reranker.model    = MagicMock(predict=MagicMock(return_value=[2.0]))
    api_module.llm.generate      = MagicMock(return_value=FAKE_ANSWER)
    api_module.llm.model         = "mistral"

    from fastapi.testclient import TestClient
    return TestClient(api_module.app), api_module


@pytest.fixture(scope="module")
def client(api):
    return api[0]


@pytest.fixture(scope="module")
def api_module(api):
    return api[1]


# ── GET / ─────────────────────────────────────────────────────────────────────

def test_root_returns_200(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_root_has_message(client):
    resp = client.get("/")
    assert "message" in resp.json()


# ── GET /health ───────────────────────────────────────────────────────────────

def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_status_ok(client):
    resp = client.get("/health")
    assert resp.json()["status"] == "ok"


def test_health_has_chunks_field(client):
    resp = client.get("/health")
    assert "chunks" in resp.json()


# ── POST /query ───────────────────────────────────────────────────────────────

def test_query_returns_200(client):
    resp = client.post("/query", json={"query": "What is attention?", "top_k": 3})
    assert resp.status_code == 200


def test_query_response_has_answer(client):
    resp = client.post("/query", json={"query": "What is attention?"})
    assert "answer" in resp.json()
    assert resp.json()["answer"] == FAKE_ANSWER


def test_query_response_has_citations(client):
    resp = client.post("/query", json={"query": "transformer"})
    data = resp.json()
    assert "citations" in data
    assert isinstance(data["citations"], list)


def test_query_response_has_papers(client):
    resp = client.post("/query", json={"query": "transformer"})
    assert "papers" in resp.json()


def test_query_response_has_sources_used(client):
    resp = client.post("/query", json={"query": "transformer"})
    assert "sources_used" in resp.json()


def test_query_response_has_latency_ms(client):
    resp = client.post("/query", json={"query": "transformer"})
    data = resp.json()
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], int)


def test_query_response_has_cited_answer(client):
    resp = client.post("/query", json={"query": "What is attention?"})
    assert resp.status_code == 200
    assert "cited_answer" in resp.json()
    assert isinstance(resp.json()["cited_answer"], str)


def test_query_response_has_faithfulness_score(client):
    resp = client.post("/query", json={"query": "What is attention?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "faithfulness_score" in data
    assert 0.0 <= data["faithfulness_score"] <= 1.0


def test_query_empty_string_returns_400(client):
    resp = client.post("/query", json={"query": "   "})
    assert resp.status_code == 400


def test_query_missing_query_field_returns_422(client):
    resp = client.post("/query", json={"top_k": 5})
    assert resp.status_code == 422


def test_query_default_top_k_searches_double(client, api_module):
    api_module.retriever.search.reset_mock()
    client.post("/query", json={"query": "test"})
    call = api_module.retriever.search.call_args
    k_used = call[0][1] if len(call[0]) > 1 else call[1].get("k")
    assert k_used == 10  # top_k default=5 → search k=5*2=10


# ── POST /ask ─────────────────────────────────────────────────────────────────

def test_ask_returns_200(client):
    resp = client.post("/ask", json={"question": "What is RAG?"})
    assert resp.status_code == 200


def test_ask_response_has_answer(client):
    resp = client.post("/ask", json={"question": "What is RAG?"})
    data = resp.json()
    assert "answer" in data
    assert data["answer"] == FAKE_ANSWER


def test_ask_empty_question_returns_400(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 400


# ── Citation structure ────────────────────────────────────────────────────────

def test_citations_have_required_fields(client):
    resp = client.post("/query", json={"query": "attention"})
    for citation in resp.json()["citations"]:
        assert "title" in citation
        assert "source" in citation
        assert "chunk_id" in citation
        assert "text" in citation


def test_papers_deduplicated_by_source(client):
    resp = client.post("/query", json={"query": "attention"})
    papers = resp.json()["papers"]
    sources = [p["source"] for p in papers]
    assert len(sources) == len(set(sources))
