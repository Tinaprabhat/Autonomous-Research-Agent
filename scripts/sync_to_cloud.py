"""
sync_to_cloud.py — copy the local embedded Qdrant collection (vectors + payload) to Qdrant Cloud /
a Docker Qdrant server WITHOUT re-embedding, then verify payload indexes and HNSW config.

    python scripts/sync_to_cloud.py              # dry run: shows what would happen, writes nothing
    python scripts/sync_to_cloud.py --yes        # actually write to the remote in QDRANT_URL

Reads QDRANT_URL / QDRANT_API_KEY (or the legacy Quadrant_* spellings) from .env. Refuses to touch a
remote collection that already holds points unless --replace is given. Needs the local store to be
free (no API server / ingestion process running).
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="perform the write (default is a dry run)")
    ap.add_argument("--replace", action="store_true", help="delete and recreate a non-empty remote collection")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    from qdrant_client import models
    from backend.app import config
    # [DEAD CODE] unused import `point_id` (checked 2026-10-15)
    from backend.app.storage.qdrant_store import QdrantStore, make_client

    if not config.QDRANT_URL:
        sys.exit("QDRANT_URL is not set — nothing to sync to.")
    local = QdrantStore(make_client(path=config.QDRANT_PATH))
    n_local = local.count()
    if n_local == 0:
        sys.exit(f"Local collection '{local.collection}' at {config.QDRANT_PATH} is empty.")

    remote = QdrantStore(make_client(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, force_local=False))
    exists = remote.client.collection_exists(remote.collection)
    n_remote = remote.count() if exists else 0
    host = config.QDRANT_URL.split("//")[-1].split("/")[0]
    print(f"local : {n_local} points ({config.QDRANT_PATH})")
    print(f"remote: {n_remote} points in '{remote.collection}' on {host}")
    if n_remote and not a.replace:
        sys.exit("Remote collection is not empty. Re-run with --replace to overwrite it.")
    if not a.yes:
        print("\nDRY RUN — nothing written. Re-run with --yes to copy.")
        return

    if exists and a.replace:
        remote.client.delete_collection(remote.collection)
    remote.ensure_collection()          # HNSW m/ef_construct + payload indexes
    offset, copied = None, 0
    while True:
        pts, offset = local.client.scroll(local.collection, limit=a.batch, offset=offset,
                                          with_payload=True, with_vectors=True)
        if not pts:
            break
        remote.client.upsert(remote.collection, wait=True, points=[
            models.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in pts])
        copied += len(pts)
        print(f"  copied {copied}/{n_local}", end="\r", flush=True)
        if offset is None:
            break
    print()
    remote.verify_indexes()             # raises if any payload index / HNSW setting is missing
    got = remote.count()
    if got != n_local:
        sys.exit(f"MISMATCH: remote has {got} points, expected {n_local}")
    print(f"OK — {got} points on remote; payload indexes and HNSW config verified via collection_info.")


if __name__ == "__main__":
    main()
