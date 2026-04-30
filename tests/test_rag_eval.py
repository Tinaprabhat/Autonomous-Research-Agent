"""
RAGWatch evaluation — runs as part of the normal pytest suite.
Uses the same 50 QA pairs as the RAGAS evaluation for a fair comparison.

Run:
    pytest tests/test_rag_eval.py -v
"""
import json, os, sys
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ragwatch.pytest_plugin.api import evaluate_rag, evaluate_one, Thresholds
from tests.evaluation.eval_ragas_50 import QA_PAIRS   # reuse same 50 pairs

# ── convert to ragwatch dict format ──────────────────────────────────────────
# ragwatch case_from_dict accepts: query, context (list[str]), answer, ground_truth

RW_CASES = [
    {
        "query":        p["query"],
        "context":      p["context"],
        "answer":       p["answer"],
        "ground_truth": p["reference"],
    }
    for p in QA_PAIRS
]


# ── Test 1: aggregate quality over all 50 cases ───────────────────────────────

def test_rag_overall_quality():
    """Aggregate over all 50 cases: composite >= 0.45, no hallucination check per-case."""
    report = evaluate_rag(
        RW_CASES,
        thresholds=Thresholds(
            composite=0.45,       # below the observed mean of 0.60 — catches real regressions
            faithfulness=None,    # checked per-case in test_individual_case_faithfulness
            hallucination_max=None,
        ),
        mode="cpu_safe",
    )
    out_path = os.path.join(os.path.dirname(__file__), "evaluation", "ragwatch_50_report.json")
    _save_report(report, out_path)
    assert report.passed, report.summary()


# ── Test 2: per-case spot checks (parametrized on first 10 pairs) ─────────────

@pytest.mark.parametrize("case", RW_CASES[:10], ids=[p["query"][:40] for p in QA_PAIRS[:10]])
def test_individual_case_faithfulness(case):
    """Each of the first 10 cases should have faithfulness >= 0.3 (NLI-based)."""
    result = evaluate_one(case, mode="cpu_safe")
    score = result.faithfulness.score if result.faithfulness else 0.0
    assert score >= 0.3, (
        f"Query: {case['query']}\n"
        f"Faithfulness: {score:.3f} < 0.3\n"
        f"Answer: {case['answer'][:100]}"
    )


# ── Test 3: no case should have extreme hallucination ────────────────────────

@pytest.mark.parametrize("case", RW_CASES, ids=[p["query"][:35] for p in QA_PAIRS])
def test_no_severe_hallucination(case):
    """Hallucination score should stay below 0.75 for all 50 cases."""
    result = evaluate_one(case, mode="cpu_safe")
    hallu = result.hallucination_score.score if result.hallucination_score else 0.0
    assert hallu < 0.75, (
        f"Query: {case['query']}\n"
        f"Hallucination: {hallu:.3f} >= 0.75\n"
        f"Answer: {case['answer'][:100]}"
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_report(report, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "n_cases":            report.n_cases,
        "passed":             report.passed,
        "composite_mean":     report.composite_mean,
        "composite_std":      report.composite_std,
        "faithfulness_mean":  report.faithfulness_mean,
        "hallucination_mean": report.hallucination_mean,
        "n_failures":         len(report.failures),
        "failures":           report.failures[:20],
        "per_case": [
            {
                "label":        label,
                "composite":    r.composite.score     if r.composite     else None,
                "faithfulness": r.faithfulness.score  if r.faithfulness  else None,
                "hallucination":r.hallucination_score.score if r.hallucination_score else None,
                "answer_relevance": r.answer_relevance.score if r.answer_relevance else None,
                "completeness": r.completeness.score  if r.completeness  else None,
            }
            for label, r in zip(report.case_labels, report.per_case)
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
