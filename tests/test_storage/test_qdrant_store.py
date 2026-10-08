"""Unit tests for backend/app/storage/qdrant_store.py (in-memory Qdrant)."""
import numpy as np
import pytest

from backend.app.storage.qdrant_store import (
    PAYLOAD_INDEXES, QdrantStore, make_client, point_id)


def test_point_id_is_deterministic_and_unique():
    assert point_id("a_v1_0") == point_id("a_v1_0")
    assert point_id("a_v1_0") != point_id("b_v1_0")


def test_make_client_with_path_is_always_local(monkeypatch):
    monkeypatch.setattr("backend.app.config.QDRANT_URL", "https://example.invalid:6333")
    store = QdrantStore(make_client(path=":memory:"))
    assert store._is_local


def test_count_and_search(mem_store, fake_embedder):
    assert mem_store.count() == 10
    hits = mem_store.search(fake_embedder.embed_query("self-attention transformers"), 3)
    assert len(hits) == 3
    assert "score" in hits[0] and "arxiv_id" in hits[0]


def test_upsert_is_idempotent(mem_store, fake_embedder):
    chunks = mem_store.scroll_all()
    mem_store.add_embeddings(fake_embedder.embed_documents([c["text"] for c in chunks]), chunks)
    assert mem_store.count() == 10


def test_filter_by_category(mem_store, fake_embedder):
    hits = mem_store.search(fake_embedder.embed_query("models"), 10, {"category": "cs.LG"})
    assert hits and all(h["category"] == "cs.LG" for h in hits)


def test_filter_by_author_list_field(mem_store, fake_embedder):
    hits = mem_store.search(fake_embedder.embed_query("models"), 10, {"author": "Ada Lovelace"})
    assert hits and all("Ada Lovelace" in h["author"] for h in hits)


def test_filter_by_date_range(mem_store, fake_embedder):
    hits = mem_store.search(fake_embedder.embed_query("models"), 10,
                            {"date_from": "2024-02-01", "date_to": "2024-02-28"})
    assert hits and all(h["published_date"].startswith("2024-02") for h in hits)


def test_filter_multiple_values_and_combination(mem_store, fake_embedder):
    hits = mem_store.search(fake_embedder.embed_query("models"), 10,
                            {"section": ["Method"], "category": "cs.CL"})
    assert all(h["section"] == "Method" and h["category"] == "cs.CL" for h in hits)


def test_filter_no_match_returns_empty(mem_store, fake_embedder):
    assert mem_store.search(fake_embedder.embed_query("x"), 5, {"arxiv_id": "0000.00000"}) == []


def test_scroll_all_applies_server_side_filter(mem_store):
    rows = mem_store.scroll_all({"arxiv_id": "2401.00000"})
    assert {r["arxiv_id"] for r in rows} == {"2401.00000"}


def test_delete_paper(mem_store):
    mem_store.delete_paper("2401.00000")
    assert all(r["arxiv_id"] != "2401.00000" for r in mem_store.scroll_all())


def test_search_on_missing_collection_returns_empty():
    s = QdrantStore(make_client(path=":memory:"), collection="nope")
    assert s.search(np.zeros(384), 5) == []
    assert s.count() == 0


def test_length_mismatch_raises(mem_store):
    with pytest.raises(ValueError):
        mem_store.add_embeddings(np.zeros((2, 384)), [{"chunk_id": "x"}])


def test_expected_payload_indexes_declared():
    assert set(PAYLOAD_INDEXES) == {"arxiv_id", "category", "published_date", "author", "section"}


def test_verify_indexes_fails_loudly_when_missing():
    """Simulate a server collection with no payload indexes."""
    from types import SimpleNamespace
    s = QdrantStore(make_client(path=":memory:"), collection="t")
    s._is_local = False  # pretend to be a real server
    info = SimpleNamespace(payload_schema={"arxiv_id": 1},
                           config=SimpleNamespace(hnsw_config=None))
    s.client = SimpleNamespace(get_collection=lambda name: info)
    with pytest.raises(RuntimeError, match="Missing Qdrant payload indexes"):
        s.verify_indexes()


def test_verify_indexes_passes_when_all_present():
    from types import SimpleNamespace
    from backend.app.storage.qdrant_store import HNSW_EF_CONSTRUCT, HNSW_M
    s = QdrantStore(make_client(path=":memory:"), collection="t")
    s._is_local = False
    info = SimpleNamespace(
        payload_schema={k: 1 for k in PAYLOAD_INDEXES},
        config=SimpleNamespace(hnsw_config=SimpleNamespace(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT)))
    s.client = SimpleNamespace(get_collection=lambda name: info)
    s.verify_indexes()
