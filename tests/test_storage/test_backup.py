"""Backup/restore verification logic, offline (in-memory Qdrant). The real snapshot round-trip is the
`scripts/qdrant_backup.py drill` against a Qdrant server; its last result is docs/restore_drill_latest.json."""
import json

import pytest
from qdrant_client import models

from backend.app.storage import backup
from backend.app.storage.qdrant_store import QdrantStore, make_client


def build_pair(mem_store, fake_embedder):
    """Source = the fixture store; target = an exact copy in a second collection of the same client."""
    src = mem_store
    pts, _ = src.client.scroll(src.collection, limit=100, with_payload=True, with_vectors=True)
    dst = QdrantStore(src.client, collection="copy")
    dst.ensure_collection()
    dst.client.upsert("copy", points=[models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in pts])
    src.ensure_collection()
    return src, dst


def test_identical_copy_passes_every_check(mem_store, fake_embedder):
    src, dst = build_pair(mem_store, fake_embedder)
    res = backup.compare_collections(src.client, src.collection, dst.collection, sample=50, queries=5)
    assert res["ok"], res
    assert set(res["checks"]) == {"point_count", "vector_config", "hnsw_config", "payload_indexes",
                                  "sampled_points_identical", "search_results_identical"}


def test_missing_points_are_detected(mem_store, fake_embedder):
    src, dst = build_pair(mem_store, fake_embedder)
    victim = dst.client.scroll("copy", limit=1)[0][0].id
    dst.client.delete("copy", points_selector=models.PointIdsList(points=[victim]))
    res = backup.compare_collections(src.client, src.collection, dst.collection, sample=50, queries=5)
    assert not res["ok"] and not res["checks"]["point_count"]["ok"]


def test_corrupted_payload_is_detected(mem_store, fake_embedder):
    src, dst = build_pair(mem_store, fake_embedder)
    pid = dst.client.scroll("copy", limit=1)[0][0].id
    dst.client.set_payload("copy", payload={"section": "TAMPERED"}, points=[pid])
    res = backup.compare_collections(src.client, src.collection, dst.collection, sample=50, queries=5)
    assert not res["ok"] and not res["checks"]["sampled_points_identical"]["ok"]


def test_corrupted_vector_is_detected(mem_store, fake_embedder):
    src, dst = build_pair(mem_store, fake_embedder)
    p = dst.client.scroll("copy", limit=1, with_vectors=True)[0][0]
    flipped = [-x for x in p.vector]
    dst.client.update_vectors("copy", points=[models.PointVectors(id=p.id, vector=flipped)])
    res = backup.compare_collections(src.client, src.collection, dst.collection, sample=50, queries=5)
    assert not res["ok"] and not res["checks"]["sampled_points_identical"]["ok"]


def test_restore_refuses_the_live_collection(mem_store, tmp_path):
    f = tmp_path / "x.snapshot"
    f.write_bytes(b"data")
    with pytest.raises(backup.BackupError, match="live collection"):
        backup.restore("http://x", None, str(f), "papers", "papers", mem_store.client)


def test_restore_refuses_a_nonempty_target(mem_store, tmp_path):
    f = tmp_path / "x.snapshot"
    f.write_bytes(b"data")
    with pytest.raises(backup.BackupError, match="already holds points"):
        backup.restore("http://x", None, str(f), mem_store.collection, "papers", mem_store.client)


def test_restore_rejects_a_snapshot_that_fails_its_checksum(mem_store, tmp_path):
    f = tmp_path / "x.snapshot"
    f.write_bytes(b"original")
    (tmp_path / "x.snapshot.json").write_text(json.dumps({"sha256": backup.sha256_file(str(f))}))
    f.write_bytes(b"tampered")                                   # file changes after the manifest was written
    with pytest.raises(backup.BackupError, match="sha256"):
        backup.restore("http://x", None, str(f), "scratch", "papers", mem_store.client)


def test_drill_never_targets_the_live_collection(mem_store, tmp_path):
    with pytest.raises(backup.BackupError):
        backup.drill(mem_store.client, "http://x", None, "papers", str(tmp_path), scratch="papers")
