"""eval/ragas_metrics.py: metric derivations, budget-error plumbing, and that RAGAS really imports (via the compat stub)."""
import pytest

from eval import generation as G
from eval import ragas_metrics as rm


@pytest.fixture(autouse=True)
def _offline_api_key(monkeypatch):
    """CI has no GROQ_API_KEY and the OpenAI client refuses to build without one; no test here makes a network call."""
    from backend.app import config
    if not config.GROQ_API_KEY:
        monkeypatch.setattr(config, "GROQ_API_KEY", "test-key-not-real")


class StubJudge:
    """Stands in for RagasJudge.raw(): records what each metric asked for."""
    def __init__(self, raw):
        self._raw, self.asked = raw, []

    def raw(self, tc, needed):
        self.asked.append(set(needed))
        return self._raw


def metric(name, raw):
    j = StubJudge(raw)
    m = rm._Metric(j, name)
    m.measure(rm.EvalCase("q?", "answer", ["ctx"], "ref"))
    return m.score, j.asked


def test_f1_is_the_harmonic_mean_and_zero_safe():
    assert rm.f1(0.5, 0.5) == 0.5
    assert rm.f1(1.0, 0.0) == 0.0 and rm.f1(0.0, 0.0) == 0.0
    assert rm.f1(0.25, 1.0) == pytest.approx(0.4)


def test_hallucination_is_the_complement_of_faithfulness():
    score, asked = metric("hallucination", {"faithfulness": 0.75})
    assert score == pytest.approx(0.25) and asked == [{"faithfulness"}]
    assert metric("faithfulness", {"faithfulness": 0.75})[0] == 0.75


def test_correctness_is_f1_of_precision_and_recall_and_completeness_is_recall():
    raw = {"fact_precision": 0.25, "fact_coverage": 1.0}
    score, asked = metric("correctness", raw)
    assert score == pytest.approx(0.4) and asked == [{"fact_precision", "fact_coverage"}]
    score, asked = metric("completeness", raw)
    assert score == 1.0 and asked == [{"fact_coverage"}]          # completeness needs only the reference-coverage pass


def test_the_four_metrics_are_exposed_through_the_judge_interface():
    class J(rm.RagasJudge):
        def __init__(self):
            pass
    assert set(J().metrics()) == {"faithfulness", "hallucination", "correctness", "completeness"}


def test_wait_seconds_parses_the_groq_hint_and_has_a_default():
    assert rm._wait_seconds("Please try again in 5.355s. Need more") == pytest.approx(6.355)
    assert rm._wait_seconds("try again in 2m10.5s") == pytest.approx(131.5)
    assert rm._wait_seconds("no hint") == 20.0


def test_the_budget_error_type_is_shared_with_the_generation_loop():
    assert G.JudgeBudgetExhausted is rm.JudgeBudgetExhausted
    assert G.is_budget_error(rm.JudgeBudgetExhausted("daily cap"))


def test_cached_scores_live_on_the_case_object_not_on_its_id():
    tc1, tc2 = rm.EvalCase("q", "a", ["c"]), rm.EvalCase("q", "a", ["c"])
    tc1.__dict__.setdefault("_raw", {})["faithfulness"] = 1.0
    assert "_raw" not in tc2.__dict__                              # a different case never sees another case's cache


def _ragas_or_skip():
    """Skip only if ragas is genuinely not installed; a broken import must FAIL, not skip."""
    try:
        return rm.load_ragas()
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "ragas":
            pytest.skip("ragas is not installed")
        raise


def test_ragas_imports_in_this_environment_through_the_compat_stub():
    llm_factory, Faithfulness, FactualCorrectness = _ragas_or_skip()
    assert callable(llm_factory) and Faithfulness.__name__ == "Faithfulness" and FactualCorrectness.__name__ == "FactualCorrectness"


def test_a_real_judge_object_builds_without_network_calls():
    _ragas_or_skip()
    j = rm.RagasJudge(model="openai/gpt-oss-120b")
    assert set(j.metrics()) == {"faithfulness", "hallucination", "correctness", "completeness"} and j.tokens == 0


def test_consecutive_scoring_calls_share_one_event_loop():
    """Regression: a fresh asyncio.run() per call broke the HTTP client ("Event loop is closed") from the 2nd call on."""
    _ragas_or_skip()
    j = rm.RagasJudge(model="openai/gpt-oss-120b")
    loops = []

    async def fake_compute(tc, needed):
        import asyncio
        loops.append(asyncio.get_running_loop())
        return {"faithfulness": 1.0}

    j._compute = fake_compute
    tc = rm.EvalCase("q", "a", ["c"])
    j.raw(tc, {"faithfulness"})
    j.raw(tc, {"faithfulness"})
    assert len(loops) == 2 and loops[0] is loops[1] and not loops[0].is_closed()
    j.close()


def test_a_daily_quota_error_wrapped_by_ragas_is_still_recognised():
    """Regression: instructor wrapped the 429, so the evaluation crashed (exit 1) instead of stopping cleanly (exit 3)."""
    inner = rm.JudgeBudgetExhausted("daily")
    try:
        try:
            raise inner
        except Exception as e:
            raise RuntimeError("InstructorRetryException wrapper") from e
    except RuntimeError as outer:
        assert rm.is_daily_budget_error(outer)
    assert rm.is_daily_budget_error(RuntimeError("Rate limit ... on tokens per day (TPD): Limit 200000"))
    assert not rm.is_daily_budget_error(ValueError("bad json"))
    assert not rm.is_daily_budget_error(RuntimeError("on tokens per minute (TPM): Limit 8000"))


def test_raw_converts_a_wrapped_daily_error_into_judge_budget_exhausted():
    _ragas_or_skip()
    j = rm.RagasJudge(model="openai/gpt-oss-120b")

    async def boom(tc, needed):
        try:
            raise RuntimeError("Rate limit reached on tokens per day (TPD)")
        except RuntimeError as e:
            raise ValueError("instructor retry") from e

    j._compute = boom
    with pytest.raises(rm.JudgeBudgetExhausted):
        j.raw(rm.EvalCase("q", "a", ["c"]), {"faithfulness"})
    j.close()


def test_precision_and_recall_come_from_one_pair_of_claim_passes():
    """Running precision and recall as two RAGAS metrics repeated the response-vs-reference pass (6 calls); one pair is enough (4)."""
    import numpy as np

    class FakeFC:
        calls = []
        def __init__(self, llm=None, mode="f1"):
            pass
        async def _decompose_and_verify_claims(self, text, other):
            FakeFC.calls.append((text, other))
            # response claims vs reference: 3 supported, 1 not; reference claims vs response: 2 supported, 2 not
            return np.array([True, True, True, False]) if text == "RESP" else np.array([True, True, False, False])

    class J(rm.RagasJudge):
        def __init__(self):
            self._FactualCorrectness = FakeFC
            self.llm = None
    tc = rm.EvalCase("q", "RESP", ["c"], "REF")
    import asyncio
    got = asyncio.run(J()._compute(tc, {"fact_precision", "fact_coverage", "fact_recall"}))
    # tp=3, fp=1, fn=2 -> precision 3/4, coverage = 2 of 4 reference claims supported = 0.5, RAGAS recall tp/(tp+fn) = 3/5
    assert got == {"fact_precision": 0.75, "fact_coverage": 0.5, "fact_recall": 0.6}
    assert FakeFC.calls == [("RESP", "REF"), ("REF", "RESP")]                   # exactly two passes
    assert tc._raw == got                                                       # cached on the case: no second computation
    asyncio.run(J()._compute(tc, {"fact_coverage"}))
    assert len(FakeFC.calls) == 2


def test_f1_matches_ragas_when_built_from_those_two_numbers():
    assert rm.f1(0.75, 0.6) == pytest.approx(2 * 0.75 * 0.6 / 1.35)


def _fake_judge(first, second):
    """A judge whose claim passes return fixed arrays: first = response-vs-reference, second = reference-vs-response."""
    import numpy as np

    class FakeFC:
        calls = []
        def __init__(self, llm=None, mode="f1"):
            pass
        async def _decompose_and_verify_claims(self, text, other):
            FakeFC.calls.append(text)
            return np.array(first if text == "RESP" else second, dtype=bool)

    class J(rm.RagasJudge):
        def __init__(self):
            self._FactualCorrectness = FakeFC
            self.llm = None
    return J(), FakeFC


def test_coverage_is_nonzero_when_the_response_covers_the_reference_even_if_no_response_claim_is_supported():
    """Regression for the artefact found on 2026-10-15: a correct answer that restates the question's context was scored
    completeness 0.0 because RAGAS recall is tp/(tp+fn) with tp = response claims the reference supports."""
    import asyncio
    j, fc = _fake_judge(first=[False], second=[True])           # the one response claim is 'unsupported'; the reference claim IS covered
    tc = rm.EvalCase("q", "RESP", ["c"], "REF")
    got = asyncio.run(j._compute(tc, {"fact_precision", "fact_coverage", "fact_recall"}))
    assert got["fact_precision"] == 0.0 and got["fact_coverage"] == 1.0 and got["fact_recall"] == 0.0


def test_only_the_passes_that_are_needed_are_run():
    import asyncio
    j, fc = _fake_judge(first=[True], second=[True, False])
    asyncio.run(j._compute(rm.EvalCase("q", "RESP", ["c"], "REF"), {"fact_coverage"}))
    assert fc.calls == ["REF"]                                    # coverage alone needs only the reference-vs-response pass


def test_an_empty_claim_list_is_a_failed_measurement_not_a_zero():
    import asyncio
    j, _ = _fake_judge(first=[], second=[True])
    with pytest.raises(ValueError):
        asyncio.run(j._compute(rm.EvalCase("q", "RESP", ["c"], "REF"), {"fact_precision"}))
