"""
ingest_papers.py — harvest arXiv papers and ingest them into Qdrant.

    python scripts/ingest_papers.py harvest [--max-papers 100] [--queries ...] [--categories ...]
    python scripts/ingest_papers.py run [--limit N]          # download->parse->chunk->embed
    python scripts/ingest_papers.py status
    python scripts/ingest_papers.py retry-failed
    python scripts/ingest_papers.py verify                   # fails loudly if indexes missing

Target store is chosen by env: QDRANT_URL set -> remote (Docker / Qdrant Cloud),
otherwise embedded local storage at QDRANT_PATH.  Add --local to force local.
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DEFAULT_QUERIES = [
    "meta-learning few-shot learning",
    "retrieval-augmented generation",
    "dense passage retrieval",
    "in-context learning large language models",
    "parameter-efficient fine-tuning LoRA",
    "hallucination detection language models",
    "transformer self-attention",
    "prompt injection attacks language models",
]
DEFAULT_CATEGORIES = ["cs.LG", "cs.CL", "cs.IR", "cs.AI"]
MANIFEST_PATH = "data/manifest.json"


def cmd_harvest(args, queue):
    from backend.app.ingestion.arxiv_fetcher import harvest
    papers = harvest(args.queries, args.max_papers, args.categories)
    res = {"added": 0, "duplicate": 0, "upgraded": 0}
    for p in papers:
        res[queue.enqueue(p)] += 1
    print(f"Harvested {len(papers)} papers: {res}")
    export_manifest(queue)


def cmd_run(args, queue):
    from tqdm import tqdm
    from backend.app.ingestion.worker import IngestionWorker
    from backend.app.retrieval.embedder import get_embedder
    from backend.app.storage.qdrant_store import QdrantStore

    store = QdrantStore()
    worker = IngestionWorker(queue, store, get_embedder())
    todo = queue.runnable(args.limit)
    bar = tqdm(total=len(todo), desc="Ingesting")
    counts = worker.run(args.limit, progress=lambda aid, st: (bar.update(1), bar.set_postfix_str(f"{aid}:{st}")))
    bar.close()
    print("Queue:", counts)
    print(f"Points in collection '{store.collection}': {store.count()}")
    for f in queue.failed():
        print(f"  FAILED {f['arxiv_id']}{f['version']}: {f['error_msg']}")
    cmd_verify(args, queue, store)
    export_manifest(queue)


def cmd_verify(args, queue, store=None):
    from backend.app.storage.qdrant_store import QdrantStore
    store = store or QdrantStore()
    if store.count() == 0:
        print("verify: collection is empty, nothing to check")
        return
    if store._is_local:
        print("verify: SKIPPED — embedded local Qdrant has no server-side payload indexes. "
              "Run against Docker / Qdrant Cloud (set QDRANT_URL) to verify them.")
        return
    store.verify_indexes()    # raises RuntimeError if any payload index / HNSW setting is missing
    print("verify: OK — all payload indexes + HNSW config confirmed via collection_info")


def cmd_status(args, queue):
    print(json.dumps(queue.counts(), indent=2))
    for f in queue.failed():
        print(f"FAILED {f['arxiv_id']}{f['version']} (retries={f['retry_count']}): {f['error_msg']}")


def cmd_retry(args, queue):
    print(f"Reset {queue.reset_failed()} failed jobs to pending")


def export_manifest(queue):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(queue.manifest(), f, indent=2)
    print(f"Manifest -> {MANIFEST_PATH}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="force embedded local Qdrant")
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest")
    h.add_argument("--max-papers", type=int, default=100)
    h.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    h.add_argument("--categories", nargs="*", default=DEFAULT_CATEGORIES)
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int, default=None)
    sub.add_parser("status")
    sub.add_parser("retry-failed")
    sub.add_parser("verify")
    args = ap.parse_args()

    if args.local:
        os.environ["QDRANT_FORCE_LOCAL"] = "1"
    from backend.app.ingestion.job_queue import JobQueue
    queue = JobQueue()
    {"harvest": cmd_harvest, "run": cmd_run, "status": cmd_status,
     "retry-failed": cmd_retry, "verify": cmd_verify}[args.cmd](args, queue)


if __name__ == "__main__":
    main()
