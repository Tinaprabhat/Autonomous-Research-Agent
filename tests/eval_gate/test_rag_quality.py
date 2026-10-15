"""
RAGAS regression gate.

    pytest -m eval tests/eval_gate

Runs the live pipeline (Qdrant + Groq) on the golden set's `regression` subset, scores each guarded response with RAGAS
(faithfulness, correctness, completeness; see eval/ragas_metrics.py), asserts per-case floors, then a composite gate.
Skipped automatically when keys or a reviewed golden set are unavailable, so it never breaks the offline unit-test job.
Thresholds can be tuned with env vars: GATE_FAITHFULNESS, GATE_CORRECTNESS, GATE_COMPLETENESS, GATE_COMPOSITE.
The defaults are provisional floors, not measured targets: see info.md section 11.8 for the scores they should be tightened to.
"""
import os
import statistics

import pytest

from eval import golden

pytestmark = pytest.mark.eval

T_FAITH = float(os.getenv("GATE_FAITHFULNESS", "0.7"))
T_CORRECT = float(os.getenv("GATE_CORRECTNESS", "0.3"))
T_COMPLETE = float(os.getenv("GATE_COMPLETENESS", "0.5"))
T_COMPOSITE = float(os.getenv("GATE_COMPOSITE", "0.5"))

try:
    CASES = [c for c in golden.load(only_reviewed=True, regression_only=True) if c.answerable]
except (OSError, golden.GoldenSetError):
    CASES = []

_needs = pytest.mark.skipif(
    not (os.getenv("GROQ_API_KEY") and CASES),
    reason="needs GROQ_API_KEY and reviewed regression cases in eval/data/golden_set.json")

_SCORES: list[dict] = []


@pytest.fixture(scope="module")
def pipeline():
    from backend.app.services import build_pipeline
    p, _ = build_pipeline(use_rewriter=False)
    return p


@pytest.fixture(scope="module")
def judge():
    from eval.ragas_metrics import RagasJudge
    return RagasJudge()


@_needs
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_case(case, pipeline, judge):
    from eval.ragas_metrics import EvalCase

    res = pipeline.answer(case.question, top_k=5)
    assert res.status != "rejected", "benign golden question was rejected by the input guard"
    ctx = [c["text"] for c in res.citations]
    tc = EvalCase(question=case.question, response=res.answer, contexts=ctx, reference=case.reference_answer or "")
    scores = {}
    for name, floor in (("faithfulness", T_FAITH), ("correctness", T_CORRECT), ("completeness", T_COMPLETE)):
        m = judge.metrics()[name]
        m.measure(tc)
        scores[name] = m.score
    _SCORES.append(scores)
    for name, floor in (("faithfulness", T_FAITH), ("correctness", T_CORRECT), ("completeness", T_COMPLETE)):
        assert scores[name] >= floor, f"{case.id}: {name} {scores[name]:.2f} < {floor}"


@_needs
def test_zz_composite_gate():
    """Fails the build when the mean of the three headline metrics drops below the threshold."""
    assert _SCORES, "no per-case scores were collected"
    comp = statistics.mean(statistics.mean(s.values()) for s in _SCORES)
    assert comp >= T_COMPOSITE, f"composite {comp:.3f} < gate {T_COMPOSITE}"
