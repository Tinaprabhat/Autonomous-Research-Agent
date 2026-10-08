"""
worker.py — drives jobs through download -> parse -> chunk -> embed (Qdrant).

Resumable: each job restarts from the stage recorded in the queue. Downloader,
parser, chunker and embedder are injectable so the flow is testable offline.
"""
from __future__ import annotations

import json
import logging
import os

from backend.app import config
from backend.app.ingestion.chunking import chunk_segments
from backend.app.ingestion.job_queue import JobQueue
from backend.app.ingestion.pdf_parser import parse_pdf

log = logging.getLogger(__name__)

CHUNK_DIR = "data/processed_chunks"


class IngestionWorker:

    def __init__(self, queue: JobQueue, store, embedder, downloader=None,
                 parser=parse_pdf, chunker=chunk_segments, chunk_dir: str = CHUNK_DIR,
                 raw_dir: str | None = None):
        self.queue, self.store, self.embedder = queue, store, embedder
        if downloader is None:
            from backend.app.ingestion.arxiv_fetcher import download_pdf
            downloader = download_pdf
        self.downloader, self.parser, self.chunker = downloader, parser, chunker
        self.chunk_dir = chunk_dir
        self.raw_dir = raw_dir or config.RAW_DIR

    # ── one job ──────────────────────────────────────────────────────────────

    def _chunks_path(self, job: dict) -> str:
        safe = job["arxiv_id"].replace("/", "_")
        return os.path.join(self.chunk_dir, f"{safe}_{job['version']}.json")

    def process(self, job: dict) -> str:
        """Advance one job as far as possible. Returns its final status."""
        aid = job["arxiv_id"]
        meta = json.loads(job["meta_json"])
        status = job["status"]
        try:
            if status == "pending":
                path = self.downloader(meta, self.raw_dir)
                self.queue.advance(aid, "downloaded", pdf_path=path)
                job["pdf_path"], status = path, "downloaded"

            if status in ("downloaded", "parsed"):
                parsed = self.parser(job["pdf_path"])
                if not parsed.segments:
                    raise ValueError("no extractable text")
                self.queue.advance(aid, "parsed")
                paper = {k: meta[k] for k in
                         ("arxiv_id", "version", "title", "author", "category",
                          "published_date") if k in meta}
                paper["used_ocr"] = parsed.used_ocr
                chunks = self.chunker(parsed.segments, paper)
                if not chunks:
                    raise ValueError("chunking produced no chunks")
                os.makedirs(self.chunk_dir, exist_ok=True)
                cpath = self._chunks_path(job)
                with open(cpath, "w", encoding="utf-8") as f:
                    json.dump(chunks, f)
                self.queue.advance(aid, "chunked", chunks_path=cpath, n_chunks=len(chunks))
                job["chunks_path"], status = cpath, "chunked"

            if status == "chunked":
                with open(job["chunks_path"], encoding="utf-8") as f:
                    chunks = json.load(f)
                self.store.ensure_collection()
                self.store.delete_paper(aid)          # replace any older version's points
                vecs = self.embedder.embed_documents([c["text"] for c in chunks])
                self.store.add_embeddings(vecs, chunks)
                self.queue.advance(aid, "embedded")
                status = "embedded"
            return status
        except Exception as e:
            log.warning("job %s failed at %s: %s", aid, status, e)
            return self.queue.fail(aid, f"{type(e).__name__}: {e}")

    def run(self, limit: int | None = None, progress=None) -> dict:
        """Process runnable jobs until none remain (or `limit` jobs started)."""
        jobs = self.queue.runnable(limit)
        for j in jobs:
            # a job may need several passes if it keeps failing at the same stage
            while True:
                st = self.process(j)
                if st in ("embedded", "failed"):
                    break
                fresh = [x for x in self.queue.runnable() if x["arxiv_id"] == j["arxiv_id"]]
                if not fresh or fresh[0]["retry_count"] == j["retry_count"]:
                    break
                j = fresh[0]
            if progress:
                progress(j["arxiv_id"], st)
        return self.queue.counts()
