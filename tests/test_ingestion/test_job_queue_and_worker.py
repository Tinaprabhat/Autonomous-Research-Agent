"""Tests for job_queue.py, worker.py and arxiv_fetcher helpers (all offline)."""
import json

import pytest

from backend.app.ingestion.arxiv_fetcher import split_id_version
from backend.app.ingestion.job_queue import MAX_RETRIES, JobQueue
from backend.app.ingestion.pdf_parser import ParsedPdf, Segment
from backend.app.ingestion.worker import IngestionWorker

TEXT = ("Self-attention lets every token attend to every other token in the sequence, "
        "which makes long-range dependencies easy to model in transformers.")


def meta(aid="2401.00001", v="v1"):
    return {"arxiv_id": aid, "version": v, "title": f"Paper {aid}", "author": ["A"],
            "category": "cs.LG", "published_date": "2024-01-01T00:00:00Z"}


@pytest.fixture
def q():
    return JobQueue(":memory:")


# ── helpers ───────────────────────────────────────────────────────────────────

def test_split_id_version():
    assert split_id_version("2101.00001v2") == ("2101.00001", "v2")
    assert split_id_version("2101.00001") == ("2101.00001", "v1")
    assert split_id_version("cs.LG/0701001v3") == ("cs.LG/0701001", "v3")


# ── queue ─────────────────────────────────────────────────────────────────────

def test_enqueue_and_dedup(q):
    assert q.enqueue(meta()) == "added"
    assert q.enqueue(meta()) == "duplicate"
    assert q.counts()["pending"] == 1


def test_newer_version_requeues(q):
    q.enqueue(meta(v="v1"))
    q.advance("2401.00001", "embedded")
    assert q.enqueue(meta(v="v2")) == "upgraded"
    row = q.runnable()[0]
    assert row["status"] == "pending" and row["version"] == "v2"


def test_older_version_is_duplicate(q):
    q.enqueue(meta(v="v3"))
    assert q.enqueue(meta(v="v2")) == "duplicate"


def test_fail_retries_then_permanent(q):
    q.enqueue(meta())
    for i in range(MAX_RETRIES - 1):
        assert q.fail("2401.00001", "boom") == "pending"
    assert q.fail("2401.00001", "boom") == "failed"
    assert q.failed()[0]["error_msg"] == "boom"
    assert q.reset_failed() == 1 and q.counts()["pending"] == 1


def test_manifest_lists_all_papers(q):
    q.enqueue(meta("1.1")); q.enqueue(meta("2.2"))
    assert [m["arxiv_id"] for m in q.manifest()] == ["1.1", "2.2"]


# ── worker ────────────────────────────────────────────────────────────────────

def _parsed(*a):
    segs = [Segment(TEXT, 1, "Method", 0, len(TEXT))]
    return ParsedPdf(TEXT, segs, 1)


def make_worker(q, store, emb, tmp_path, **kw):
    return IngestionWorker(
        q, store, emb,
        downloader=kw.get("downloader", lambda m, d: str(tmp_path / "x.pdf")),
        parser=kw.get("parser", _parsed),
        chunker=kw.get("chunker", lambda segs, paper: [
            {"text": TEXT, "chunk_id": f"{paper['arxiv_id']}_{paper['version']}_0", **paper}]),
        chunk_dir=str(tmp_path / "chunks"))


def test_worker_happy_path_embeds_into_store(q, mem_store, fake_embedder, tmp_path):
    mem_store.client.delete_collection(mem_store.collection)
    q.enqueue(meta("1111.1111"))
    counts = make_worker(q, mem_store, fake_embedder, tmp_path).run()
    assert counts["embedded"] == 1 and counts["failed"] == 0
    rows = mem_store.scroll_all({"arxiv_id": "1111.1111"})
    assert len(rows) == 1 and rows[0]["version"] == "v1"


def test_worker_records_failure_and_continues(q, mem_store, fake_embedder, tmp_path):
    q.enqueue(meta("bad.0001")); q.enqueue(meta("good.0001"))

    def parser(path):
        if "bad" in str(path):
            raise ValueError("corrupt pdf")
        return _parsed()
    paths = {}
    def dl(m, d):
        paths[m["arxiv_id"]] = str(tmp_path / (m["arxiv_id"] + ".pdf"))
        return paths[m["arxiv_id"]]
    counts = make_worker(q, mem_store, fake_embedder, tmp_path, parser=parser, downloader=dl).run()
    assert counts["failed"] == 1 and counts["embedded"] == 1
    assert "corrupt pdf" in q.failed()[0]["error_msg"]


def test_worker_resumes_from_chunked_stage_without_redownload(q, mem_store, fake_embedder, tmp_path):
    q.enqueue(meta("3333.3333"))
    chunks = [{"text": TEXT, "chunk_id": "3333.3333_v1_0", "arxiv_id": "3333.3333", "version": "v1"}]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(chunks))
    q.advance("3333.3333", "chunked", chunks_path=str(p), n_chunks=1)

    def no_download(m, d):
        raise AssertionError("must not re-download")
    counts = make_worker(q, mem_store, fake_embedder, tmp_path, downloader=no_download).run()
    assert counts["embedded"] == 1


def test_worker_no_text_is_a_failure(q, mem_store, fake_embedder, tmp_path):
    q.enqueue(meta())
    w = make_worker(q, mem_store, fake_embedder, tmp_path,
                    parser=lambda p: ParsedPdf("", [], 1))
    w.run()
    assert q.counts()["failed"] == 1


def test_new_version_replaces_old_points(q, mem_store, fake_embedder, tmp_path):
    mem_store.client.delete_collection(mem_store.collection)
    q.enqueue(meta("4444.4444", "v1"))
    w = make_worker(q, mem_store, fake_embedder, tmp_path)
    w.run()
    q.enqueue(meta("4444.4444", "v2"))
    w.run()
    rows = mem_store.scroll_all({"arxiv_id": "4444.4444"})
    assert {r["version"] for r in rows} == {"v2"}
