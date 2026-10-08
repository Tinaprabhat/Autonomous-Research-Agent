"""
relabel_sections.py — re-derive `section` labels for already-ingested papers WITHOUT re-embedding.

Use after changing section-labelling rules in ingestion/pdf_parser.py. Only labels change (never which
blocks are kept), so text offsets are identical and each stored chunk is matched to its new label by
(char_start, text). Chunk ids and vectors are untouched; only the `section` payload field is updated.

    python scripts/relabel_sections.py --local [--dry-run]
"""
import argparse
import collections
# [DEAD CODE] unused import (checked 2026-10-08)
# import glob
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--raw-dir", default="data/raw_papers")
    ap.add_argument("--chunk-dir", default="data/processed_chunks")
    a = ap.parse_args()
    if a.local:
        os.environ["QDRANT_FORCE_LOCAL"] = "1"

    from qdrant_client import models
    from backend.app.ingestion.pdf_parser import parse_pdf
    from backend.app.storage.qdrant_store import QdrantStore, point_id

    store = QdrantStore()
    rows = store.scroll_all()
    by_paper = collections.defaultdict(list)
    for r in rows:
        by_paper[(r["arxiv_id"], r["version"])].append(r)

    before, after = collections.Counter(), collections.Counter()
    changed = skipped = 0
    for (aid, ver), chunks in sorted(by_paper.items()):
        pdf = os.path.join(a.raw_dir, f"{aid.replace('/', '_')}_{ver}.pdf")
        if not os.path.exists(pdf):
            skipped += len(chunks)
            print(f"  [skip] no PDF for {aid}{ver}")
            continue
        parsed = parse_pdf(pdf)
        updates = {}
        for c in chunks:
            seg = next((s for s in parsed.segments if s.start <= c["char_start"] < s.end + 2), None)
            # Labels depend on position only; multi-block chunks don't always byte-match their span.
            new = seg.section if seg is not None else c["section"]
            if seg is None:
                skipped += 1
            before[c["section"]] += 1
            after[new] += 1
            if new != c["section"]:
                updates[c["chunk_id"]] = new
        changed += len(updates)
        if updates and not a.dry_run:
            for sec in set(updates.values()):
                ids = [point_id(cid) for cid, s in updates.items() if s == sec]
                store.client.set_payload(store.collection, payload={"section": sec}, points=ids)
            cpath = os.path.join(a.chunk_dir, f"{aid.replace('/', '_')}_{ver}.json")
            if os.path.exists(cpath):
                with open(cpath, encoding="utf-8") as f:
                    cj = json.load(f)
                for c in cj:
                    c["section"] = updates.get(c["chunk_id"], c["section"])
                with open(cpath, "w", encoding="utf-8") as f:
                    json.dump(cj, f)

    total = sum(before.values())
    print(f"\n{'DRY RUN: ' if a.dry_run else ''}relabelled {changed} of {total} chunks ({skipped} skipped)")
    for name in ("Conclusion", "Appendix", "Introduction", "Method"):
        print(f"  {name:<13} {before[name]:>6} ({100*before[name]/total:4.1f}%) -> {after[name]:>6} ({100*after[name]/total:4.1f}%)")


if __name__ == "__main__":
    main()
