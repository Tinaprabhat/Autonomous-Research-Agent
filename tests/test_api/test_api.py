"""API tests with an injected fake pipeline (no models, no network)."""
import json

import pytest
from fastapi.testclient import TestClient

from backend.app import config
from backend.app.api import RateLimiter, create_app
from backend.app.pipeline import PipelineResult


class FakePipeline:
    class llm:
        model = "fake-model"

    def __init__(self):
        self.calls = []

    def answer(self, q, top_k=5, filters=None):
        self.calls.append((q, top_k, filters))
        return PipelineResult(status="ok", answer="An answer [1].", citations=[{"n": 1}], sources_used=1)

    def answer_stream(self, q, top_k=5, filters=None):
        self.calls.append((q, top_k, filters))
        yield {"event": "sources", "citations": []}
        yield {"event": "token", "text": "An "}
        yield {"event": "final", "result": self.answer(q).to_dict()}


class FakeStore:
    def count(self): return 3
    def list_papers(self):
        return [{"arxiv_id": "2401.1", "title": "T", "category": "cs.LG"}]


@pytest.fixture
def pipe():
    return FakePipeline()


@pytest.fixture
def client(pipe):
    return TestClient(create_app(pipeline=pipe, store=FakeStore(), rate_limit_per_min=1000))


def test_health(client):
    j = client.get("/health").json()
    assert j == {"status": "ok", "chunks": 3, "model": "fake-model"}


def test_query_returns_pipeline_result(client):
    r = client.post("/query", json={"query": "what is lora?"})
    assert r.status_code == 200 and r.json()["answer"] == "An answer [1]."
    assert r.json()["status"] == "ok"


def test_filters_are_forwarded(client, pipe):
    client.post("/query", json={"query": "q", "top_k": 3,
                                "filters": {"category": "cs.LG", "date_from": "2024-01-01"}})
    assert pipe.calls[-1] == ("q", 3, {"category": "cs.LG", "date_from": "2024-01-01"})


def test_empty_filters_become_none(client, pipe):
    client.post("/query", json={"query": "q", "filters": {}})
    assert pipe.calls[-1][2] is None


def test_blank_query_rejected(client):
    assert client.post("/query", json={"query": "   "}).status_code == 400
    assert client.post("/query", json={"query": ""}).status_code == 422


def test_query_length_cap(client):
    r = client.post("/query", json={"query": "a" * (config.MAX_QUERY_CHARS + 1)})
    assert r.status_code == 422


def test_top_k_bounds(client):
    assert client.post("/query", json={"query": "q", "top_k": 0}).status_code == 422
    assert client.post("/query", json={"query": "q", "top_k": 99}).status_code == 422


def test_bad_date_format_rejected(client):
    r = client.post("/query", json={"query": "q", "filters": {"date_from": "yesterday"}})
    assert r.status_code == 422


def test_stream_is_sse(client):
    with client.stream("POST", "/query/stream", json={"query": "q"}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    assert [e["event"] for e in events] == ["sources", "token", "final"]


def test_stream_errors_become_error_event(pipe):
    def boom(*a, **k):
        raise RuntimeError("kaput")
        yield
    pipe.answer_stream = boom
    c = TestClient(create_app(pipeline=pipe, store=FakeStore(), rate_limit_per_min=1000))
    with c.stream("POST", "/query/stream", json={"query": "q"}) as r:
        events = [json.loads(l[6:]) for l in r.iter_lines() if l.startswith("data: ")]
    assert events[-1]["event"] == "error"
    assert "kaput" not in json.dumps(events)        # no internals leaked


def test_legacy_ask(client):
    assert client.post("/ask", json={"question": "q"}).json()["answer"] == "An answer [1]."


def test_papers_endpoint(client):
    j = client.get("/papers").json()
    assert j["categories"] == ["cs.LG"] and "Method" in j["sections"]


def test_feedback_is_logged(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FEEDBACK_LOG", str(tmp_path / "fb.jsonl"))
    r = client.post("/feedback", json={"query": "q", "kind": "wrong_citation", "citation": 2})
    assert r.status_code == 200
    rec = json.loads((tmp_path / "fb.jsonl").read_text())
    assert rec["kind"] == "wrong_citation" and rec["citation"] == 2


def test_feedback_validates_kind(client):
    assert client.post("/feedback", json={"kind": "lol"}).status_code == 422


def test_metrics_endpoint(client):
    assert "latency_ms_p95" in client.get("/metrics").json()


def test_index_serves_frontend(client):
    r = client.get("/")
    assert r.status_code == 200


def test_503_while_loading():
    c = TestClient(create_app(pipeline=None, rate_limit_per_min=1000))   # no lifespan -> not loaded
    assert c.post("/query", json={"query": "q"}).status_code == 503


# ── rate limiting ────────────────────────────────────────────────────────────

def test_rate_limit_per_ip(pipe):
    c = TestClient(create_app(pipeline=pipe, store=FakeStore(), rate_limit_per_min=3))
    codes = [c.post("/query", json={"query": "q"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert c.post("/query", json={"query": "q"}).headers["retry-after"] == "60"


def test_rate_limit_is_per_ip(pipe):
    c = TestClient(create_app(pipeline=pipe, store=FakeStore(), rate_limit_per_min=1))
    h1, h2 = {"x-forwarded-for": "1.1.1.1"}, {"x-forwarded-for": "2.2.2.2"}
    assert c.post("/query", json={"query": "q"}, headers=h1).status_code == 200
    assert c.post("/query", json={"query": "q"}, headers=h1).status_code == 429
    assert c.post("/query", json={"query": "q"}, headers=h2).status_code == 200


def test_health_not_rate_limited(pipe):
    c = TestClient(create_app(pipeline=pipe, store=FakeStore(), rate_limit_per_min=1))
    assert all(c.get("/health").status_code == 200 for _ in range(10))


def test_limiter_window_slides():
    rl = RateLimiter(2)
    assert rl.allow("k", 0) and rl.allow("k", 1) and not rl.allow("k", 2)
    assert rl.allow("k", 61)
