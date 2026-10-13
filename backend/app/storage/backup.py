"""
backup.py - snapshot, download, restore and verify a Qdrant collection (Qdrant Cloud or Docker server).

Why this exists: a Qdrant Cloud *free* cluster has no managed backups and pauses when idle, so the only way to
survive losing the cluster is an off-cluster copy of the data plus a restore path that has actually been run.

    snapshot   -> server-side snapshot of the live collection, downloaded to disk with a sha256 + manifest
    restore    -> upload the snapshot into a NEW collection name (never the live one unless forced)
    verify     -> compare the restored collection with the source: count, config, indexes, vectors, search results
    drill      -> snapshot + restore into a scratch collection + verify + delete scratch (the tested restore)

Snapshots are server-side only: the embedded local Qdrant mode cannot create them.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from urllib.parse import quote

import numpy as np
import requests

from backend.app.storage.qdrant_store import HNSW_EF_CONSTRUCT, HNSW_M, PAYLOAD_INDEXES

CHUNK = 1024 * 1024


class BackupError(RuntimeError):
    pass


def _headers(api_key: str | None) -> dict:
    return {"api-key": api_key} if api_key else {}


def _base(url: str) -> str:
    return url.rstrip("/")


# -- snapshot -----------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def create_and_download(client, url: str, api_key: str | None, collection: str, out_dir: str,
                        timeout: int = 600) -> dict:
    """Create a snapshot on the server, stream it to `out_dir`, write <file>.json with sha256 + metadata."""
    os.makedirs(out_dir, exist_ok=True)
    points = client.count(collection, exact=True).count
    t0 = time.time()
    snap = client.create_snapshot(collection_name=collection, wait=True)
    if snap is None:
        raise BackupError("server did not return a snapshot description")
    name = snap.name
    dest = os.path.join(out_dir, f"{collection}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{name}")
    r = requests.get(f"{_base(url)}/collections/{quote(collection)}/snapshots/{quote(name)}",
                     headers=_headers(api_key), stream=True, timeout=timeout)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for block in r.iter_content(CHUNK):
            f.write(block)
    size = os.path.getsize(dest)
    if size == 0 or (snap.size and size != snap.size):
        os.remove(dest)
        raise BackupError(f"downloaded snapshot size {size} != server-reported {snap.size}")
    meta = {
        "collection": collection, "server_snapshot_name": name, "file": os.path.basename(dest),
        "bytes": size, "sha256": sha256_file(dest), "points_at_snapshot": points,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1),
    }
    with open(dest + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def prune_remote_snapshots(client, collection: str, keep: int) -> list[str]:
    """Delete all but the newest `keep` server-side snapshots (they live on the cluster's own disk)."""
    snaps = sorted(client.list_snapshots(collection), key=lambda s: s.creation_time or "")
    doomed = snaps[:-keep] if keep > 0 else snaps
    for s in doomed:
        client.delete_snapshot(collection, s.name, wait=True)
    return [s.name for s in doomed]


# -- restore ------------------------------------------------------------------

def restore(url: str, api_key: str | None, snapshot_file: str, target: str, live_collection: str,
            client, force_live: bool = False, timeout: int = 1800) -> dict:
    """Upload a snapshot into collection `target`. Refuses the live collection and any non-empty target."""
    if target == live_collection and not force_live:
        raise BackupError(f"refusing to restore over the live collection '{live_collection}' "
                          "(restore to a new name, verify, then switch; or pass --force-live)")
    meta_path = snapshot_file + ".json"
    if os.path.exists(meta_path):
        expected = json.load(open(meta_path, encoding="utf-8")).get("sha256")
        if expected and sha256_file(snapshot_file) != expected:
            raise BackupError("snapshot file does not match its recorded sha256 (corrupted or tampered)")
    if client.collection_exists(target):
        if client.count(target, exact=True).count and not force_live:
            raise BackupError(f"target collection '{target}' already holds points")
        client.delete_collection(target)
    t0 = time.time()
    with open(snapshot_file, "rb") as f:
        r = requests.post(f"{_base(url)}/collections/{quote(target)}/snapshots/upload",
                          params={"priority": "snapshot", "wait": "true"},
                          headers=_headers(api_key), files={"snapshot": (os.path.basename(snapshot_file), f)},
                          timeout=timeout)
    if r.status_code >= 300:
        raise BackupError(f"restore upload failed: HTTP {r.status_code} {r.text[:300]}")
    return {"target": target, "seconds": round(time.time() - t0, 1)}


# -- verify -------------------------------------------------------------------

def _vec(p) -> np.ndarray:
    v = p.vector
    return np.asarray(v if not isinstance(v, dict) else next(iter(v.values())), dtype="float32")


def _same_ranking(a, b, tol: float = 1e-5) -> bool:
    """Same scores in the same order, and the same ids wherever the score is not tied with a neighbour."""
    if len(a) != len(b):
        return False
    sa, sb = [p.score for p in a], [p.score for p in b]
    if any(abs(x - y) > tol for x, y in zip(sa, sb)):
        return False
    for i, (pa, pb) in enumerate(zip(a, b)):
        # a score equal to the cutoff may tie with points that fell just outside the top-k
        tied = (i > 0 and abs(sa[i] - sa[i - 1]) <= tol) or (i + 1 < len(sa) and abs(sa[i] - sa[i + 1]) <= tol)             or abs(sa[i] - sa[-1]) <= tol
        if not tied and pa.id != pb.id:
            return False
    return True


def compare_collections(client, source: str, target: str, sample: int = 200, queries: int = 20,
                        top_k: int = 5, seed: int = 7) -> dict:
    """Checks that `target` is a faithful copy of `source`. Returns {checks: {...}, ok: bool}."""
    checks: dict[str, dict] = {}

    def check(name, ok, detail=""):
        checks[name] = {"ok": bool(ok), "detail": detail}

    n_src = client.count(source, exact=True).count
    n_dst = client.count(target, exact=True).count
    check("point_count", n_src == n_dst, f"source={n_src} target={n_dst}")

    a, b = client.get_collection(source), client.get_collection(target)
    va, vb = a.config.params.vectors, b.config.params.vectors
    check("vector_config", (va.size, va.distance) == (vb.size, vb.distance),
          f"{va.size}/{va.distance} vs {vb.size}/{vb.distance}")
    ha, hb = a.config.hnsw_config, b.config.hnsw_config
    check("hnsw_config", (ha.m, ha.ef_construct) == (hb.m, hb.ef_construct) == (HNSW_M, HNSW_EF_CONSTRUCT),
          f"source m={ha.m}/ef={ha.ef_construct} target m={hb.m}/ef={hb.ef_construct}")
    if client._client.__class__.__name__ == "QdrantLocal":
        check("payload_indexes", True, "skipped: embedded mode ignores payload indexes")
    else:
        have = set((b.payload_schema or {}).keys())
        missing = [f for f in PAYLOAD_INDEXES if f not in have]
        check("payload_indexes", not missing, f"missing={missing}" if missing else f"{sorted(have)}")

    # exact vector + payload equality on a random sample of ids
    ids, offset = [], None
    while True:
        pts, offset = client.scroll(source, limit=1000, offset=offset, with_payload=False, with_vectors=False)
        ids.extend(p.id for p in pts)
        if offset is None:
            break
    rng = random.Random(seed)
    picked = rng.sample(ids, min(sample, len(ids)))
    src = {p.id: p for p in client.retrieve(source, picked, with_payload=True, with_vectors=True)}
    dst = {p.id: p for p in client.retrieve(target, picked, with_payload=True, with_vectors=True)}
    bad_vec = bad_payload = missing_pts = 0
    for pid in picked:
        if pid not in dst:
            missing_pts += 1
            continue
        if not np.allclose(_vec(src[pid]), _vec(dst[pid]), rtol=0, atol=1e-6):
            bad_vec += 1
        if src[pid].payload != dst[pid].payload:
            bad_payload += 1
    check("sampled_points_identical", missing_pts == bad_vec == bad_payload == 0,
          f"sampled={len(picked)} missing={missing_pts} vector_mismatch={bad_vec} payload_mismatch={bad_payload}")

    # behavioural check: same top-k ids for real queries (use sampled points' own vectors as queries)
    same = 0
    qpts = picked[:queries]
    for pid in qpts:
        qv = _vec(src[pid]).tolist()
        ra = client.query_points(source, query=qv, limit=top_k).points
        rb = client.query_points(target, query=qv, limit=top_k).points
        same += _same_ranking(ra, rb)
    check("search_results_identical", same == len(qpts), f"{same}/{len(qpts)} queries return the same top-{top_k}")

    return {"ok": all(c["ok"] for c in checks.values()), "checks": checks}


# -- drill --------------------------------------------------------------------

def drill(client, url: str, api_key: str | None, collection: str, out_dir: str,
          scratch: str | None = None, keep_snapshots: int = 3, keep_scratch: bool = False) -> dict:
    """Snapshot -> restore into a scratch collection -> verify -> delete scratch. Never writes to `collection`."""
    scratch = scratch or f"{collection}_restore_drill"
    if scratch == collection:
        raise BackupError("scratch collection must differ from the live one")
    report: dict = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "source": collection, "scratch": scratch}
    try:
        report["snapshot"] = create_and_download(client, url, api_key, collection, out_dir)
        path = os.path.join(out_dir, report["snapshot"]["file"])
        report["restore"] = restore(url, api_key, path, scratch, collection, client)
        report["verify"] = compare_collections(client, collection, scratch)
        report["ok"] = report["verify"]["ok"]
    finally:
        if not keep_scratch and client.collection_exists(scratch):
            client.delete_collection(scratch)
            report["scratch_deleted"] = True
        report["pruned_remote_snapshots"] = prune_remote_snapshots(client, collection, keep_snapshots)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return report
