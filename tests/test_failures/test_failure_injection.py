"""
Failure-injection tests: the system must degrade visibly and safely, never crash, hang or fabricate.

  1. LLM down        every generation provider fails (Groq errors / Ollama not running)
  2. Qdrant down     connection refused, and a server that accepts but never answers (timeout)
  3. Malformed PDFs  garbage bytes, truncated, empty, HTML-as-PDF, encrypted, blank (no text layer)

These use the REAL components wherever cheap (real GroqLLM wrapper, real QdrantClient pointed at a dead
port, real parse_pdf on real broken files) and only fake the network/models.
"""
import socket
import threading
import time

import fitz
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from backend.app.api import create_app
from backend.app.ingestion.chunking import chunk_segments
from backend.app.ingestion.job_queue import MAX_RETRIES, JobQueue
from backend.app.ingestion.pdf_parser import parse_pdf
from backend.app.ingestion.worker import IngestionWorker
from backend.app.llm.groq_llm import GroqLLM, LLMError
from backend.app.llm.local_llm import OllamaLLM
from backend.app.llm.router import LLMRouter
from backend.app.pipeline import UNCITED_WARNING, ResearchPipeline, RetrievalUnavailable
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.storage.qdrant_store import QdrantStore

Q = "LoRA low-rank decomposition trainable parameters"
GOOD_ANSWER = "LoRA reduces trainable parameters by injecting low-rank decomposition matrices [1]."


class IdentityReranker:
    def rerank(self, q, docs, top_k=5):
        return docs[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# 1. LLM down
# ══════════════════════════════════════════════════════════════════════════════

class _BrokenGroqClient:
    """Mimics the Groq SDK client when the API is unreachable."""
    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                raise ConnectionError("simulated: Groq API unreachable")


def dead_router():
    """Real router + real GroqLLM wrapper (broken client) + real OllamaLLM at a closed port."""
    groq = GroqLLM(client=_BrokenGroqClient())
    ollama = OllamaLLM(url="http://127.0.0.1:9/api/generate", timeout=2)
    return LLMRouter(primary=groq, fallback=ollama, extra=[])


def make_pipeline(mem_store, fake_embedder, fake_verifier, tmp_path, llm):
    return ResearchPipeline(HybridRetriever(mem_store, fake_embedder), IdentityReranker(), fake_verifier, llm,
                            log_path=str(tmp_path / "q.jsonl"))


def test_router_with_every_provider_down_raises_one_clear_llm_error():
    with pytest.raises(LLMError) as ei:
        dead_router().generate("hi")
    msg = str(ei.value)
    assert "groq" in msg and "ollama" in msg.lower()


def test_pipeline_with_llm_down_returns_unverified_extractive_answer(mem_store, fake_embedder, fake_verifier, tmp_path):
    p = make_pipeline(mem_store, fake_embedder, fake_verifier, tmp_path, dead_router())
    r = p.answer(Q, top_k=3)
    assert r.status == "unverified" and r.warning == UNCITED_WARNING
    assert r.answer and "llm_error" in r.guardrail          # shows retrieved passages, labelled unverified
    assert r.citations and r.sources_used == 3              # sources still listed so the user can read them


def test_streaming_with_llm_down_ends_with_unverified_final(mem_store, fake_embedder, fake_verifier, tmp_path):
    p = make_pipeline(mem_store, fake_embedder, fake_verifier, tmp_path, dead_router())
    events = list(p.answer_stream(Q, top_k=3))
    assert events[-1]["event"] == "final" and events[-1]["result"]["status"] == "unverified"
    assert [e["event"] for e in events].count("retry") == 1  # exactly one retry, then a visible failure


def test_llm_down_through_the_http_api_is_a_200_with_warning_not_a_500(mem_store, fake_embedder, fake_verifier, tmp_path):
    p = make_pipeline(mem_store, fake_embedder, fake_verifier, tmp_path, dead_router())
    c = TestClient(create_app(pipeline=p, store=mem_store, api_keys="", rate_limit_per_min=100))
    r = c.post("/query", json={"query": Q})
    assert r.status_code == 200 and r.json()["status"] == "unverified" and r.json()["warning"]


def test_fallback_provider_serves_a_verified_answer_when_primary_is_down(mem_store, fake_embedder, fake_verifier, tmp_path):
    class Healthy:
        model = "healthy"
        available = True

        def generate(self, prompt, use_cache=True):
            return GOOD_ANSWER

    router = LLMRouter(primary=GroqLLM(client=_BrokenGroqClient()), fallback=Healthy(), extra=[])
    r = make_pipeline(mem_store, fake_embedder, fake_verifier, tmp_path, router).answer(Q, top_k=3)
    assert r.status == "ok" and r.provider == "ollama"      # primary failed, fallback slot served it


# ══════════════════════════════════════════════════════════════════════════════
# 2. Qdrant down
# ══════════════════════════════════════════════════════════════════════════════

def dead_store(timeout=2):
    return QdrantStore(QdrantClient(url="http://127.0.0.1:9", timeout=timeout), collection="papers")


@pytest.fixture
def hung_server():
    """TCP server that accepts connections and never replies: a hung Qdrant."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    conns, stop = [], threading.Event()

    def accept():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                conns.append(srv.accept()[0])
            except OSError:
                pass

    t = threading.Thread(target=accept, daemon=True)
    t.start()
    yield srv.getsockname()[1]
    stop.set()
    t.join(timeout=2)
    for c in conns:
        c.close()
    srv.close()


def pipeline_over(store, fake_embedder, fake_verifier, tmp_path):
    class NeverCalledLLM:
        model = "never"

        def generate(self, prompt):
            raise AssertionError("LLM must not be called when nothing was retrieved")

    retr = HybridRetriever(store, fake_embedder, documents=[])      # skip the startup scroll; search hits Qdrant
    return ResearchPipeline(retr, IdentityReranker(), fake_verifier, NeverCalledLLM(),
                            log_path=str(tmp_path / "q.jsonl"))


def api_over(store, fake_embedder, fake_verifier, tmp_path):
    return TestClient(create_app(pipeline=pipeline_over(store, fake_embedder, fake_verifier, tmp_path),
                                 store=store, api_keys="", rate_limit_per_min=100))


def test_store_calls_fail_fast_when_qdrant_refuses_connections():
    with pytest.raises(Exception):
        dead_store().count()


def test_pipeline_raises_retrieval_unavailable_not_a_fabricated_answer(fake_embedder, fake_verifier, tmp_path):
    p = pipeline_over(dead_store(), fake_embedder, fake_verifier, tmp_path)
    with pytest.raises(RetrievalUnavailable):
        p.answer(Q)


def test_hung_qdrant_times_out_instead_of_blocking_forever(hung_server, fake_embedder, fake_verifier, tmp_path):
    store = QdrantStore(QdrantClient(url=f"http://127.0.0.1:{hung_server}", timeout=1), collection="papers")
    p = pipeline_over(store, fake_embedder, fake_verifier, tmp_path)
    t0 = time.time()
    with pytest.raises(RetrievalUnavailable):
        p.answer(Q)
    assert time.time() - t0 < 15


def test_api_returns_503_when_qdrant_is_down(fake_embedder, fake_verifier, tmp_path):
    c = api_over(dead_store(), fake_embedder, fake_verifier, tmp_path)
    r = c.post("/query", json={"query": Q})
    assert r.status_code == 503 and "unavailable" in r.json()["detail"].lower()
    assert c.get("/papers").status_code == 503


def test_health_reports_degraded_instead_of_crashing_the_probe(fake_embedder, fake_verifier, tmp_path):
    r = api_over(dead_store(), fake_embedder, fake_verifier, tmp_path).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded" and r.json()["chunks"] == 0


def test_stream_emits_error_event_when_qdrant_is_down(fake_embedder, fake_verifier, tmp_path):
    c = api_over(dead_store(), fake_embedder, fake_verifier, tmp_path)
    body = c.post("/query/stream", json={"query": Q}).text
    assert '"event": "error"' in body and "unavailable" in body.lower()


def test_service_recovers_once_qdrant_is_back(mem_store, fake_embedder, fake_verifier, tmp_path):
    """Same app object: fails while the DB is down, serves again when it comes back (no restart needed)."""
    class Flaky:
        def __init__(self, inner):
            self.inner, self.down = inner, True

        def __getattr__(self, name):
            if self.down:
                raise ConnectionError("simulated outage")
            return getattr(self.inner, name)

    class LLM:
        model = "ok"
        last_provider = "t"

        def generate(self, prompt):
            return GOOD_ANSWER

    flaky = Flaky(mem_store)
    retr = HybridRetriever(mem_store, fake_embedder)
    retr.vector_store = flaky
    p = ResearchPipeline(retr, IdentityReranker(), fake_verifier, LLM(), log_path=str(tmp_path / "q.jsonl"))
    c = TestClient(create_app(pipeline=p, store=flaky, api_keys="", rate_limit_per_min=100))
    assert c.post("/query", json={"query": Q}).status_code == 503
    flaky.down = False
    r = c.post("/query", json={"query": Q})
    assert r.status_code == 200 and r.json()["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# 3. Malformed PDFs
# ══════════════════════════════════════════════════════════════════════════════

GOOD_TEXT = ("Self-attention lets every token attend to every other token in the sequence, which makes "
             "long-range dependencies easy to model in transformers. " * 3)


def write_good_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 550, 780), "Introduction\n\n" + GOOD_TEXT, fontsize=11)
    doc.save(str(path))
    doc.close()


def build_bad_pdfs(tmp_path):
    good = tmp_path / "_src.pdf"
    write_good_pdf(good)
    raw = good.read_bytes()
    files = {
        "garbage": b"\x00\x01\x02 this is not a pdf at all " * 50,
        "empty": b"",
        "html_as_pdf": b"<html><body><h1>404 Not Found</h1></body></html>",
        "truncated": raw[: len(raw) // 2],
        "header_only": b"%PDF-1.7\n",
    }
    out = {}
    for name, data in files.items():
        p = tmp_path / f"{name}.pdf"
        p.write_bytes(data)
        out[name] = str(p)
    enc = tmp_path / "encrypted.pdf"
    d = fitz.open(str(good))
    d.save(str(enc), encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")
    d.close()
    out["encrypted"] = str(enc)
    blank = tmp_path / "blank.pdf"
    d = fitz.open()
    d.new_page()
    d.save(str(blank))
    d.close()
    out["blank"] = str(blank)
    return out


def chunker_with(fake_embedder):
    return lambda segs, paper: chunk_segments(
        segs, paper, embed_fn=lambda texts: fake_embedder.embed_documents(texts))


def meta(aid):
    return {"arxiv_id": aid, "version": "v1", "title": aid, "author": ["A"],
            "category": "cs.LG", "published_date": "2024-01-01T00:00:00Z"}


def test_parse_pdf_raises_a_normal_exception_on_every_malformed_file(tmp_path):
    """Each bad file must raise an Exception (caught by the worker) or yield no segments - never hang or crash."""
    for name, path in build_bad_pdfs(tmp_path).items():
        try:
            parsed = parse_pdf(path)
        except Exception:
            continue
        assert not parsed.segments, f"{name} unexpectedly produced text"


def test_parse_pdf_reads_a_wellformed_pdf(tmp_path):
    p = tmp_path / "ok.pdf"
    write_good_pdf(p)
    parsed = parse_pdf(str(p))
    assert parsed.segments and "Self-attention" in parsed.text


def test_worker_quarantines_every_malformed_pdf_and_still_ingests_the_good_one(mem_store, fake_embedder, tmp_path):
    bad = build_bad_pdfs(tmp_path)
    good = tmp_path / "good.pdf"
    write_good_pdf(good)
    paths = {**{f"bad.{k}": v for k, v in bad.items()}, "good.0001": str(good)}

    q = JobQueue(":memory:")
    for aid in paths:
        q.enqueue(meta(aid))
    mem_store.client.delete_collection(mem_store.collection)
    w = IngestionWorker(q, mem_store, fake_embedder, downloader=lambda m, d: paths[m["arxiv_id"]],
                        chunker=chunker_with(fake_embedder), chunk_dir=str(tmp_path / "chunks"))
    counts = w.run()                                        # must terminate, must not raise

    assert counts["embedded"] == 1                          # the good paper made it into the store
    assert counts["failed"] == len(bad)                     # every bad file ends 'failed', none stuck mid-queue
    assert mem_store.scroll_all({"arxiv_id": "good.0001"})
    failed = q.failed()
    assert all(f["error_msg"] for f in failed)              # each failure carries a reason
    assert all(f["retry_count"] >= MAX_RETRIES for f in failed)
    assert not mem_store.scroll_all({"arxiv_id": "bad.garbage"})    # nothing from a bad file leaked into the index


def test_failed_pdf_can_be_retried_after_the_file_is_fixed(mem_store, fake_embedder, tmp_path):
    bad = tmp_path / "fixme.pdf"
    bad.write_bytes(b"not a pdf")
    q = JobQueue(":memory:")
    q.enqueue(meta("fix.0001"))
    mem_store.client.delete_collection(mem_store.collection)
    w = IngestionWorker(q, mem_store, fake_embedder, downloader=lambda m, d: str(bad),
                        chunker=chunker_with(fake_embedder), chunk_dir=str(tmp_path / "chunks"))
    assert w.run()["failed"] == 1
    write_good_pdf(bad)                                     # a re-download produced a valid file
    assert q.reset_failed() == 1
    assert w.run()["embedded"] == 1
