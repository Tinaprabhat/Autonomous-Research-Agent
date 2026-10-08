"""
ask.py — run one question through the full pipeline from the terminal.

    python scripts/ask.py "What is MAML?" [--top-k 5] [--category cs.LG] [--local] [--json]
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--category")
    ap.add_argument("--author")
    ap.add_argument("--section")
    ap.add_argument("--date-from")
    ap.add_argument("--date-to")
    ap.add_argument("--local", action="store_true", help="force embedded local Qdrant")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.local:
        os.environ["QDRANT_FORCE_LOCAL"] = "1"

    from backend.app.services import build_pipeline
    pipeline, store = build_pipeline()
    filters = {k: v for k, v in dict(category=a.category, author=a.author, section=a.section,
                                     date_from=a.date_from, date_to=a.date_to).items() if v} or None
    res = pipeline.answer(a.question, a.top_k, filters)
    if a.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
        return
    print(f"\n[{res.status}] {res.answer}")
    if res.warning:
        print(f"  ! {res.warning}")
    print(f"  faithfulness={res.faithfulness_score}  latency={res.latency_ms}ms  "
          f"model={res.model} via {res.provider}  timings={res.timings_ms}")
    for c in res.citations:
        mark = "*" if c["cited"] else " "
        print(f" {mark}[{c['n']}] {c['title']} — {c['section']} p.{c['page']}  {c['url']}")


if __name__ == "__main__":
    main()
