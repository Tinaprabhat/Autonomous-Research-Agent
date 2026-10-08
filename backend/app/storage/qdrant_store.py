"""
qdrant_store.py — Qdrant-backed vector store (replaces the FAISS VectorStore).

Interface shape is kept from the old store: `.add_embeddings(embeddings, chunks)`
and `.search(query_embedding, k)` returning a list of chunk dicts.

Indexing
--------
* Vector field: HNSW with explicit `m` / `ef_construct` (defaults target far
  larger collections than ~40k vectors).
* Payload fields: Qdrant does NOT auto-index payload. Filtering on an
  un-indexed field degrades to a scan, so the fields we filter on are indexed
  explicitly in `ensure_collection()` and verified in `verify_indexes()`.

Embedded local mode (no URL configured) ignores payload indexes — Qdrant's local
implementation filters in-process — so `verify_indexes()` only enforces them
against a real server.
"""
from __future__ import annotations

import uuid
import warnings
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from backend.app import config

# field -> payload schema type
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "arxiv_id":       models.PayloadSchemaType.KEYWORD,
    "category":       models.PayloadSchemaType.KEYWORD,
    "published_date": models.PayloadSchemaType.DATETIME,
    "author":         models.PayloadSchemaType.KEYWORD,
    "section":        models.PayloadSchemaType.KEYWORD,
}

HNSW_M = 16
HNSW_EF_CONSTRUCT = 100

_NAMESPACE = uuid.UUID("6f1c0c0e-6a7b-4a53-9d6e-2f3a8f0b1c11")


def point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk_id so re-ingestion upserts, not duplicates."""
    return str(uuid.uuid5(_NAMESPACE, str(chunk_id)))


def make_client(url: str | None = None, api_key: str | None = None,
                path: str | None = None, force_local: bool | None = None) -> QdrantClient:
    # An explicit `path` always means embedded local storage.
    if path is not None:
        force_local = True
    url = url if url is not None else config.QDRANT_URL
    api_key = api_key if api_key is not None else config.QDRANT_API_KEY
    force_local = config.QDRANT_FORCE_LOCAL if force_local is None else force_local
    if url and not force_local:
        return QdrantClient(url=url, api_key=api_key, timeout=30)
    path = path or config.QDRANT_PATH
    if path == ":memory:":
        return QdrantClient(":memory:")
    try:
        return QdrantClient(path=path)
    except RuntimeError as e:
        if "already accessed" not in str(e):
            raise
        raise RuntimeError(
            f"Embedded Qdrant storage '{path}' is locked by another process "
            "(usually the running API server / uvicorn). Stop it before running "
            "this script, or set QDRANT_URL to use a Qdrant server for concurrent access."
        ) from e


def _iso(d: str | None) -> str | None:
    if not d:
        return None
    return d if "T" in d else f"{d}T00:00:00Z"


class QdrantStore:

    def __init__(self, client: QdrantClient | None = None,
                 collection: str | None = None, dim: int = config.EMBED_DIM):
        self.client = client or make_client()
        self.collection = collection or config.QDRANT_COLLECTION
        self.dim = dim
        self._is_local = self.client._client.__class__.__name__ == "QdrantLocal"

    # ── Collection management ────────────────────────────────────────────────

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dim, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(
                    m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
            )
        with warnings.catch_warnings():
            # embedded local mode warns that payload indexes are no-ops; expected there
            warnings.simplefilter("ignore", UserWarning)
            for field, schema in PAYLOAD_INDEXES.items():
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )

    def verify_indexes(self) -> None:
        """Raise if an expected payload index / HNSW setting is missing (server only)."""
        if self._is_local:
            return
        info = self.client.get_collection(self.collection)
        have = set((info.payload_schema or {}).keys())
        missing = [f for f in PAYLOAD_INDEXES if f not in have]
        if missing:
            raise RuntimeError(f"Missing Qdrant payload indexes: {missing}")
        hnsw = info.config.hnsw_config
        if hnsw is not None and (hnsw.m != HNSW_M or hnsw.ef_construct != HNSW_EF_CONSTRUCT):
            raise RuntimeError(
                f"HNSW config drifted: m={hnsw.m} ef_construct={hnsw.ef_construct}")

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    # ── Writes ───────────────────────────────────────────────────────────────

    def add_embeddings(self, embeddings, chunks: list[dict], batch_size: int = 128) -> None:
        vecs = np.asarray(embeddings, dtype="float32")
        if len(vecs) != len(chunks):
            raise ValueError("embeddings and chunks length mismatch")
        self.ensure_collection()
        for i in range(0, len(chunks), batch_size):
            pts = [
                models.PointStruct(id=point_id(c["chunk_id"]),
                                   vector=v.tolist(), payload=c)
                for c, v in zip(chunks[i:i + batch_size], vecs[i:i + batch_size])
            ]
            self.client.upsert(self.collection, points=pts, wait=True)

    def delete_paper(self, arxiv_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key="arxiv_id", match=models.MatchValue(value=arxiv_id))
            ])),
        )

    # ── Reads ────────────────────────────────────────────────────────────────

    @staticmethod
    def build_filter(filters: dict | None) -> models.Filter | None:
        """
        filters keys (all optional): arxiv_id, category, author, section
        (str or list[str]), date_from / date_to (ISO date strings).
        """
        if not filters:
            return None
        must: list[Any] = []
        for key in ("arxiv_id", "category", "author", "section"):
            val = filters.get(key)
            if not val:
                continue
            if isinstance(val, (list, tuple, set)):
                must.append(models.FieldCondition(
                    key=key, match=models.MatchAny(any=list(val))))
            else:
                must.append(models.FieldCondition(
                    key=key, match=models.MatchValue(value=val)))
        d_from, d_to = filters.get("date_from"), filters.get("date_to")
        if d_from or d_to:
            must.append(models.FieldCondition(
                key="published_date",
                range=models.DatetimeRange(gte=_iso(d_from), lte=_iso(d_to))))
        return models.Filter(must=must) if must else None

    def search(self, query_embedding, k: int, filters: dict | None = None) -> list[dict]:
        if not self.client.collection_exists(self.collection):
            return []
        res = self.client.query_points(
            self.collection,
            query=np.asarray(query_embedding, dtype="float32").tolist(),
            query_filter=self.build_filter(filters),
            limit=k,
            with_payload=True,
        ).points
        out = []
        for p in res:
            d = dict(p.payload or {})
            d["score"] = float(p.score)
            out.append(d)
        return out

    def scroll_all(self, filters: dict | None = None, batch: int = 512,
                   fields: list[str] | None = None) -> list[dict]:
        """All payloads matching `filters` (server-side filtered); `fields` limits payload size."""
        if not self.client.collection_exists(self.collection):
            return []
        out, offset = [], None
        flt = self.build_filter(filters)
        while True:
            pts, offset = self.client.scroll(
                self.collection, scroll_filter=flt, limit=batch,
                offset=offset, with_payload=fields if fields else True, with_vectors=False)
            out.extend(dict(p.payload or {}) for p in pts)
            if offset is None:
                break
        return out

    def list_papers(self) -> list[dict]:
        """One summary row per paper (for the filter UI)."""
        seen: dict[str, dict] = {}
        for p in self.scroll_all():
            aid = p.get("arxiv_id")
            if aid and aid not in seen:
                seen[aid] = {k: p.get(k) for k in
                             ("arxiv_id", "title", "category", "published_date", "author")}
        return list(seen.values())
