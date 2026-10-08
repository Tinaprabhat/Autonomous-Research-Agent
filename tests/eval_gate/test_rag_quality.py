"""
DeepEval regression gate (plan 9.6).

    deepeval test run tests/eval_gate/test_rag_quality.py      # or: pytest -m eval tests/eval_gate

Runs the live pipeline (Qdrant + Groq) on the golden set's `regression` subset and asserts
per-case DeepEval metrics, then a composite-threshold gate. Skipped automatically when keys
or a reviewed golden set are unavailable, so it never breaks the offline unit-test job.
Thresholds can be tuned with env vars: GATE_FAITHFULNESS, GATE_RELEVANCY, GATE_CITATION, GATE_COMPOSITE.
"""
import os
import statistics

import pytest

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from eval import golden
from eval.runner import citation_metric

pytestmark = pytest.mark.eval

T_FAITH = float(os.getenv("GATE_FAITHFULNESS", "0.7"))
T_REL = float(os.getenv("GATE_RELEVANCY", "0.6"))
T_CITE = float(os.getenv("GATE_CITATION", "0.6"))
T_COMPOSITE = float(os.getenv("GATE_COMPOSITE", "0.7"))

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
    from eval.judge import GroqJudge
    return GroqJudge()


@_needs
@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_case(case, pipeline, judge):
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    res = pipeline.answer(case.question, top_k=5)
    assert res.status != "rejected", "benign golden question was rejected by the input guard"
    ctx = [c["text"] for c in res.citations]
    tc = LLMTestCase(input=case.question, actual_output=res.answer,
                     expected_output=case.reference_answer or None, retrieval_context=ctx)
    faith = FaithfulnessMetric(threshold=T_FAITH, model=judge, async_mode=False, include_reason=False)
    rel = AnswerRelevancyMetric(threshold=T_REL, model=judge, async_mode=False, include_reason=False)
    cite = citation_metric(judge)
    cite.threshold = T_CITE
    for m in (faith, rel, cite):
        m.measure(tc)
    _SCORES.append({"faith": faith.score, "rel": rel.score, "cite": cite.score})
    assert_test(tc, [faith, rel, cite])


@_needs
def test_zz_composite_gate():
    """Fails the build when the mean of the three headline metrics drops below the threshold."""
    assert _SCORES, "no per-case scores were collected"
    comp = statistics.mean(statistics.mean(s.values()) for s in _SCORES)
    assert comp >= T_COMPOSITE, f"composite {comp:.3f} < gate {T_COMPOSITE}"
