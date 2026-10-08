"""
job_queue.py — SQLite ingestion queue.

States:  pending -> downloaded -> parsed -> chunked -> embedded   (or failed)

PDF parsing / OCR failures are expected at any scale, so every paper is a row
with retry_count + error_msg, and the worker is resumable: re-running picks each
paper up at the stage it stopped. Dedup key is the *base* arXiv id; a newer
version of an already-queued paper resets it to `pending` (old points are
replaced at embed time).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager

from backend.app import config

STATES = ("pending", "downloaded", "parsed", "chunked", "embedded", "failed")
# [DEAD CODE] `NEXT` has no callers (checked 2026-10-08); commented out, restore if needed.
# NEXT = {"pending": "downloaded", "downloaded": "parsed", "parsed": "chunked",
#         "chunked": "embedded"}
MAX_RETRIES = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    arxiv_id     TEXT PRIMARY KEY,
    version      TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',
    retry_count  INTEGER NOT NULL DEFAULT 0,
    error_msg    TEXT,
    meta_json    TEXT NOT NULL,
    pdf_path     TEXT,
    chunks_path  TEXT,
    n_chunks     INTEGER,
    updated_at   REAL NOT NULL
)
"""


def _vnum(v: str) -> int:
    return int(str(v).lstrip("v") or 1)


class JobQueue:

    def __init__(self, path: str | None = None):
        self.path = path or config.JOBS_DB
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._mem = sqlite3.connect(":memory:") if self.path == ":memory:" else None
        with self._conn() as c:
            c.execute(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = self._mem or sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            if self._mem is None:
                conn.close()

    # ── enqueue / dedup ──────────────────────────────────────────────────────

    def enqueue(self, meta: dict) -> str:
        """Returns 'added' | 'duplicate' | 'upgraded' (newer version re-queued)."""
        with self._conn() as c:
            row = c.execute("SELECT version FROM jobs WHERE arxiv_id=?", (meta["arxiv_id"],)).fetchone()
            now = time.time()
            if row is None:
                c.execute(
                    "INSERT INTO jobs (arxiv_id, version, meta_json, updated_at) VALUES (?,?,?,?)",
                    (meta["arxiv_id"], meta["version"], json.dumps(meta), now))
                return "added"
            if _vnum(meta["version"]) > _vnum(row["version"]):
                c.execute(
                    "UPDATE jobs SET version=?, status='pending', retry_count=0, error_msg=NULL, "
                    "meta_json=?, pdf_path=NULL, chunks_path=NULL, n_chunks=NULL, updated_at=? "
                    "WHERE arxiv_id=?",
                    (meta["version"], json.dumps(meta), now, meta["arxiv_id"]))
                return "upgraded"
            return "duplicate"

    # ── worker interface ─────────────────────────────────────────────────────

    def runnable(self, limit: int | None = None) -> list[dict]:
        q = "SELECT * FROM jobs WHERE status NOT IN ('embedded','failed') ORDER BY updated_at"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q)]

    def advance(self, arxiv_id: str, status: str, **fields) -> None:
        assert status in STATES
        cols = {"pdf_path", "chunks_path", "n_chunks"} & fields.keys()
        sets = ["status=?", "error_msg=NULL", "updated_at=?"] + [f"{k}=?" for k in cols]
        vals = [status, time.time()] + [fields[k] for k in cols]
        with self._conn() as c:
            c.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE arxiv_id=?", (*vals, arxiv_id))

    def fail(self, arxiv_id: str, error: str) -> str:
        """Record a failure; permanent `failed` after MAX_RETRIES. Returns new status."""
        with self._conn() as c:
            row = c.execute("SELECT status, retry_count FROM jobs WHERE arxiv_id=?", (arxiv_id,)).fetchone()
            retries = row["retry_count"] + 1
            status = "failed" if retries >= MAX_RETRIES else row["status"]
            c.execute("UPDATE jobs SET status=?, retry_count=?, error_msg=?, updated_at=? WHERE arxiv_id=?",
                      (status, retries, error[:500], time.time(), arxiv_id))
            return status

    def reset_failed(self) -> int:
        with self._conn() as c:
            cur = c.execute("UPDATE jobs SET status='pending', retry_count=0, error_msg=NULL "
                            "WHERE status='failed'")
            return cur.rowcount

    # ── reporting ────────────────────────────────────────────────────────────

    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status").fetchall()
        out = {s: 0 for s in STATES}
        out.update({r["status"]: r["n"] for r in rows})
        return out

    def failed(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT arxiv_id, version, retry_count, error_msg FROM jobs WHERE status='failed'")]

    def manifest(self) -> list[dict]:
        """All papers (metadata only) — the re-ingestion recovery list."""
        with self._conn() as c:
            return [json.loads(r["meta_json"]) | {"status": r["status"]}
                    for r in c.execute("SELECT meta_json, status FROM jobs ORDER BY arxiv_id")]
