"""
eval_live_ragwatch.py — RAGWatch evaluation on LIVE pipeline outputs.

Unlike eval_ragas_50.py (which uses pre-written static answers),
this script:
  1. POSTs every query to the running API  →  gets cited_answer + real chunks
  2. Feeds those live outputs to RAGWatch  →  NLI faithfulness on actual results
  3. Saves ragwatch_live_report.json alongside the static report for comparison

Run (backend must be live on :8000):
    python -m tests.evaluation.eval_live_ragwatch
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import requests
from ragwatch.pytest_plugin.api import evaluate_rag, evaluate_one, Thresholds
from tests.evaluation.eval_ragas_50 import QA_PAIRS

API_BASE  = "http://localhost:8000"
OUT_PATH  = os.path.join(os.path.dirname(__file__), "ragwatch_live_report.json")
TOP_K     = 5
TIMEOUT_S = 120          # per-query HTTP timeout


# ── Step 1: collect live answers ──────────────────────────────────────────────

def _query(question: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/query",
            json={"query": question, "top_k": TOP_K},
            timeout=TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [WARN] query failed: {e}")
        return None


def collect_live_cases(pairs: list[dict]) -> list[dict]:
    """Query every QA pair through the live API; build RAGWatch-format cases."""
    cases       = []
    guard_scores = []

    print(f"\nQuerying live API ({API_BASE}) for {len(pairs)} questions …\n")
    for i, pair in enumerate(pairs, 1):
        q = pair["query"]
        print(f"  [{i:02d}/{len(pairs)}] {q[:60]}", end=" ", flush=True)

        t0  = time.time()
        res = _query(q)
        elapsed = time.time() - t0

        if res is None:
            print("✗ FAILED — using static fallback")
            # fall back to static answer so evaluation still runs
            cases.append({
                "query":        q,
                "context":      pair["context"],
                "answer":       pair["answer"],
                "ground_truth": pair["reference"],
                "_source":      "static_fallback",
            })
            continue

        # Prefer the citation-guard-filtered answer; fall back to raw answer
        answer  = res.get("cited_answer") or res.get("answer", "")
        # Use the actual retrieved chunk texts as context
        context = [c["text"] for c in res.get("citations", []) if c.get("text")]
        if not context:
            context = pair["context"]   # safety fallback

        guard_f = res.get("faithfulness_score")
        if guard_f is not None:
            guard_scores.append(guard_f)

        print(f"✓  ({elapsed:.1f}s)  guard_faith={guard_f:.3f}" if guard_f is not None else f"✓  ({elapsed:.1f}s)")

        cases.append({
            "query":              q,
            "context":            context,
            "answer":             answer,
            "ground_truth":       pair["reference"],
            "_guard_faithfulness": guard_f,
            "_source":            "live_pipeline",
            "_raw_answer":        res.get("answer", ""),
        })

    print(f"\nCollected {len(cases)} live cases.")
    if guard_scores:
        print(f"Citation-guard faithfulness  mean={sum(guard_scores)/len(guard_scores):.3f}  "
              f"min={min(guard_scores):.3f}  max={max(guard_scores):.3f}")
    return cases


# ── Step 2: run RAGWatch ──────────────────────────────────────────────────────

def run_ragwatch(cases: list[dict]) -> dict:
    # Strip internal _keys before passing to RAGWatch
    rw_cases = [
        {k: v for k, v in c.items() if not k.startswith("_")}
        for c in cases
    ]

    print("\nRunning RAGWatch evaluate_rag on live outputs …")
    report = evaluate_rag(
        rw_cases,
        thresholds=Thresholds(composite=0.45, faithfulness=None, hallucination_max=None),
        mode="cpu_safe",
    )
    return report


# ── Step 3: save + print ──────────────────────────────────────────────────────

def save_report(cases: list[dict], report, path: str):
    data = {
        "n_cases":            report.n_cases,
        "passed":             report.passed,
        "composite_mean":     report.composite_mean,
        "composite_std":      report.composite_std,
        "faithfulness_mean":  report.faithfulness_mean,
        "hallucination_mean": report.hallucination_mean,
        "n_failures":         len(report.failures),
        "failures":           report.failures[:20],
        "guard_faithfulness_mean": (
            round(sum(c["_guard_faithfulness"] for c in cases
                      if c.get("_guard_faithfulness") is not None)
                  / max(1, sum(1 for c in cases if c.get("_guard_faithfulness") is not None)), 3)
        ),
        "per_case": [
            {
                "label":               label,
                "source":              c.get("_source"),
                "guard_faithfulness":  c.get("_guard_faithfulness"),
                "composite":           r.composite.score           if r.composite           else None,
                "faithfulness":        r.faithfulness.score        if r.faithfulness        else None,
                "hallucination":       r.hallucination_score.score if r.hallucination_score else None,
                "answer_relevance":    r.answer_relevance.score    if r.answer_relevance    else None,
                "completeness":        r.completeness.score        if r.completeness        else None,
            }
            for label, r, c in zip(report.case_labels, report.per_case, cases)
        ],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nReport saved → {path}")
    return data


def print_summary(data: dict):
    print("\n" + "=" * 62)
    print("RAGWatch LIVE Pipeline Report")
    print("=" * 62)
    print(f"  Cases            : {data['n_cases']}")
    print(f"  Passed           : {data['passed']}")
    print(f"  Composite mean   : {data['composite_mean']:.4f}  ± {data['composite_std']:.4f}")
    print(f"  Faithfulness mean: {data['faithfulness_mean']:.4f}")
    print(f"  Hallucination    : {data['hallucination_mean']:.4f}")
    print(f"  Guard faith mean : {data['guard_faithfulness_mean']:.3f}  (citation guard score)")
    print(f"  Failures         : {data['n_failures']}")
    print("=" * 62)

    print(f"\n{'Query':<45} {'Comp':>6} {'Faith':>6} {'Hallu':>6} {'Guard':>6}")
    print("─" * 75)
    for pc in data["per_case"]:
        comp  = f"{pc['composite']:.3f}"    if pc["composite"]    is not None else "  —  "
        faith = f"{pc['faithfulness']:.3f}" if pc["faithfulness"] is not None else "  —  "
        hallu = f"{pc['hallucination']:.3f}"if pc["hallucination"]is not None else "  —  "
        guard = f"{pc['guard_faithfulness']:.3f}" if pc["guard_faithfulness"] is not None else "  —  "
        print(f"{pc['label'][:45]:<45} {comp:>6} {faith:>6} {hallu:>6} {guard:>6}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick health check
    try:
        requests.get(f"{API_BASE}/health", timeout=5).raise_for_status()
    except Exception as e:
        sys.exit(f"Backend not reachable at {API_BASE}: {e}\n"
                 "Start it with: uvicorn backend.app.api:app --reload --port 8000")

    cases  = collect_live_cases(QA_PAIRS)
    report = run_ragwatch(cases)
    data   = save_report(cases, report, OUT_PATH)
    print_summary(data)
