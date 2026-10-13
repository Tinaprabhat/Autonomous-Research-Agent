"""
qdrant_backup.py - snapshot / restore / verify the Qdrant collection in QDRANT_URL (Cloud or Docker).

    python scripts/qdrant_backup.py snapshot [--out-dir backups] [--keep 3]
    python scripts/qdrant_backup.py restore  backups/<file> --target papers_restored [--force-live]
    python scripts/qdrant_backup.py verify   --target papers_restored
    python scripts/qdrant_backup.py drill    [--keep-scratch]      # snapshot + restore to scratch + verify + clean up

`drill` is the tested restore: it never writes to the live collection and writes docs/restore_drill_latest.json.
To recover from losing the cluster: create a new cluster, set QDRANT_URL/QDRANT_API_KEY, then
`restore <file> --target papers` (the live name is allowed only when the cluster has no such collection).
Reads QDRANT_URL / QDRANT_API_KEY / QDRANT_COLLECTION from .env.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

REPORT = "docs/restore_drill_latest.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot"); s.add_argument("--out-dir", default="backups"); s.add_argument("--keep", type=int, default=3)
    r = sub.add_parser("restore"); r.add_argument("file"); r.add_argument("--target", required=True)
    r.add_argument("--force-live", action="store_true")
    v = sub.add_parser("verify"); v.add_argument("--target", required=True); v.add_argument("--sample", type=int, default=200)
    d = sub.add_parser("drill"); d.add_argument("--out-dir", default="backups"); d.add_argument("--keep-scratch", action="store_true")
    d.add_argument("--keep", type=int, default=3)
    a = ap.parse_args()

    from backend.app import config
    from backend.app.storage import backup
    from backend.app.storage.qdrant_store import make_client

    if not config.QDRANT_URL:
        sys.exit("QDRANT_URL is not set: snapshots need a Qdrant server (Cloud or Docker), not embedded mode.")
    client = make_client(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, force_local=False)
    live = config.QDRANT_COLLECTION
    host = config.QDRANT_URL.split("//")[-1].split("/")[0]
    print(f"server {host}  live collection '{live}'")

    try:
        if a.cmd == "snapshot":
            meta = backup.create_and_download(client, config.QDRANT_URL, config.QDRANT_API_KEY, live, a.out_dir)
            pruned = backup.prune_remote_snapshots(client, live, a.keep)
            print(json.dumps({**meta, "pruned_remote": pruned}, indent=2))
        elif a.cmd == "restore":
            print(json.dumps(backup.restore(config.QDRANT_URL, config.QDRANT_API_KEY, a.file, a.target, live,
                                            client, force_live=a.force_live), indent=2))
        elif a.cmd == "verify":
            res = backup.compare_collections(client, live, a.target, sample=a.sample)
            print(json.dumps(res, indent=2))
            sys.exit(0 if res["ok"] else 1)
        elif a.cmd == "drill":
            rep = backup.drill(client, config.QDRANT_URL, config.QDRANT_API_KEY, live, a.out_dir,
                               keep_snapshots=a.keep, keep_scratch=a.keep_scratch)
            os.makedirs(os.path.dirname(REPORT), exist_ok=True)
            with open(REPORT, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
            print(json.dumps(rep, indent=2))
            print("\nRESTORE DRILL", "PASSED" if rep["ok"] else "FAILED", "->", REPORT)
            sys.exit(0 if rep["ok"] else 1)
    except backup.BackupError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
