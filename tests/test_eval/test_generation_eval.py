"""Tests for eval/generation.py: checkpoint/resume, aggregation, daily-quota handling (fake metrics)."""
import json

import pytest

from backend.app.llm.groq_llm import LLMRateLimitError
from eval import generation as G
from eval import golden, runner


class FakeMetric:
    def __init__(self, score=0.8, exc=None):
        self.score, self.exc, self.calls = score, exc, 0
    def measure(self, tc):
        self.calls += 1
        if self.exc:
            raise self.exc


class Pipe:
    class R:
        def __init__(self, status, answer):
            self.status, self.answer = status, answer
    def __init__(self):
        self.asked = []
    def answer(self, q, top_k=5):
        self.asked.append(q)
        return Pipe.R("ok", "An answer [1].") if "answerable" in q else Pipe.R("abstained", "Not enough evidence")


NAMES = ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall",
         "contextual_relevancy", "citation_correctness"]


@pytest.fixture
def setup(monkeypatch):
    metrics = {n: FakeMetric(0.8) for n in NAMES}
    monkeypatch.setattr(runner, "deepeval_metrics", lambda judge: metrics)
    monkeypatch.setattr(runner, "retrieve_for_config", lambda p, c, q, k: [{"text": "ctx one"}, {"text": "ctx two"}])
    monkeypatch.setattr(runner, "raw_answer", lambda p, q, ch: "raw answer [1].")
    cases = [golden.GoldenCase(id=f"a{i}", question=f"answerable q{i}?", gold_chunk_ids=["x"], reference_answer="r")
             for i in range(1, 4)]
    cases.append(golden.GoldenCase(id="u1", question="unknown q?", answerable=False))
    return metrics, cases


def test_scores_both_configs_and_computes_contextual_once(setup, tmp_path):
    metrics, cases = setup
    out = G.eval_generation(Pipe(), cases[:1], judge=object(), checkpoint=str(tmp_path / "c.jsonl"), log=lambda *a: None)
    raw, full = out["metrics"][G.CFG_RAW], out["metrics"][G.CFG_FULL]
    assert "contextual_precision" not in raw and "contextual_precision" in full
    assert raw["faithfulness"] == full["faithfulness"] == 0.8
    assert metrics["contextual_precision"].calls == 1 and metrics["faithfulness"].calls == 2


def test_unanswerable_measures_abstention_only(setup, tmp_path):
    metrics, cases = setup
    out = G.eval_generation(Pipe(), cases, judge=object(), checkpoint=str(tmp_path / "c.jsonl"), log=lambda *a: None)
    assert out["behaviour"]["abstention_rate_on_unanswerable"] == 1.0
    assert out["behaviour"]["answer_rate_on_answerable"] == 1.0
    assert metrics["faithfulness"].calls == 6          # 3 answerable cases x 2 configs; none for u1


def test_resume_does_not_rescore_finished_cases(setup, tmp_path):
    metrics, cases = setup
    ck = str(tmp_path / "c.jsonl")
    G.eval_generation(Pipe(), cases[:2], judge=object(), checkpoint=ck, log=lambda *a: None)
    before = metrics["faithfulness"].calls
    pipe = Pipe()
    out = G.eval_generation(pipe, cases, judge=object(), checkpoint=ck, log=lambda *a: None)
    assert metrics["faithfulness"].calls == before + 2   # only a3 scored (x2 configs)
    assert "answerable q1?" not in pipe.asked and out["cases_scored"] == 4


def test_daily_judge_quota_stops_cleanly_and_keeps_finished_work(setup, tmp_path):
    metrics, cases = setup
    ck = str(tmp_path / "c.jsonl")
    state = {"n": 0}
    orig = metrics["faithfulness"].measure
    def flaky(tc):
        state["n"] += 1
        if state["n"] > 2:         # case a1 (2 configs) scored fine; a2 hits the daily quota
            raise LLMRateLimitError("TPD", retry_after=3000, daily=True)
        orig(tc)
    metrics["faithfulness"].measure = flaky
    out = G.eval_generation(Pipe(), cases, judge=object(), checkpoint=ck, log=lambda *a: None)
    assert out["stopped_early"] and "quota" in out["stopped_early"]
    assert out["cases_scored"] == 1 and list(G.load_checkpoint(ck)) == ["a1"]


def test_single_metric_failure_is_recorded_not_fatal(setup, tmp_path):
    metrics, cases = setup
    metrics["answer_relevancy"].exc = ValueError("bad json from judge")
    out = G.eval_generation(Pipe(), cases[:1], judge=object(), checkpoint=str(tmp_path / "c.jsonl"), log=lambda *a: None)
    assert out["stopped_early"] is None
    assert out["metrics"][G.CFG_FULL]["faithfulness"] == 0.8 and out["metrics"][G.CFG_FULL]["answer_relevancy"] is None


def test_is_budget_error_walks_cause_chain():
    inner = LLMRateLimitError("x", retry_after=900, daily=True)
    try:
        try:
            raise inner
        except Exception as e:
            raise RuntimeError("deepeval wrapper") from e
    except RuntimeError as outer:
        assert G.is_budget_error(outer)
    assert not G.is_budget_error(ValueError("nope"))
    assert G.is_budget_error(RuntimeError("Rate limit ... tokens per day (TPD): Limit 200000"))


def test_torn_checkpoint_line_ignored(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"case_id": "a1"}) + "\n" + '{"case_id": "a2", "sta')
    assert list(G.load_checkpoint(str(p))) == ["a1"]


def test_dump_answers_written_for_hand_labelling(setup, tmp_path):
    metrics, cases = setup
    dump = tmp_path / "answers.jsonl"
    G.eval_generation(Pipe(), cases[:1], judge=object(), checkpoint=str(tmp_path / "c.jsonl"),
                      dump_answers=str(dump), log=lambda *a: None)
    rec = json.loads(dump.read_text().splitlines()[0])
    assert rec["case_id"] == "a1" and rec["context"] == ["ctx one", "ctx two"] and "answer" in rec


# ── abstentions must not be scored as unfaithful answers ─────────────────────

def _rec(cid, status, raw_f, full_f, answerable=True):
    return {"case_id": cid, "answerable": answerable, "status": status,
            "scores": {G.CFG_RAW: {"faithfulness": raw_f}, G.CFG_FULL: {"faithfulness": full_f}} if answerable else {}}


def test_metrics_use_answered_cases_only_and_report_answer_rate():
    recs = [_rec("a1", "ok", 1.0, 1.0), _rec("a2", "ok", 0.6, 1.0),
            _rec("a3", "abstained", 1.0, 0.0),          # abstention scores 0 on the guarded side
            _rec("u1", "abstained", 0, 0, answerable=False)]
    out = G.aggregate(recs)
    assert out["metrics"][G.CFG_FULL]["faithfulness"] == 1.0 and out["metrics"][G.CFG_FULL]["n"] == 2
    assert out["metrics"][G.CFG_RAW]["faithfulness"] == 0.8          # same two cases: paired comparison
    assert out["metrics_all_cases"][G.CFG_FULL]["faithfulness"] == pytest.approx(0.667, abs=0.001)
    assert out["behaviour"]["answer_rate_on_answerable"] == pytest.approx(0.667, abs=0.001)
    assert out["behaviour"]["abstention_rate_on_unanswerable"] == 1.0


def test_rebuild_from_checkpoint_needs_no_pipeline(tmp_path):
    ck = tmp_path / "c.jsonl"
    ck.write_text(json.dumps(_rec("a1", "ok", 1.0, 0.9)) + "\n")
    out = G.rebuild_from_checkpoint(str(ck), golden_total=100)
    assert out["cases_scored"] == 1 and out["cases_planned"] == 100
    assert out["metrics"][G.CFG_FULL]["faithfulness"] == 0.9


def test_report_mode_merges_without_erasing_retrieval(tmp_path, monkeypatch):
    import sys
    rep = tmp_path / "reports"
    rep.mkdir()
    (rep / "latest.json").write_text(json.dumps({"retrieval": {"hybrid": {"chunk_recall@5": 0.9, "n": 1}}}))
    ck = tmp_path / "c.jsonl"
    ck.write_text(json.dumps(_rec("a1", "ok", 1.0, 1.0)) + "\n")
    gp = tmp_path / "g.json"
    golden.save([golden.GoldenCase(id="a1", question="q?", gold_chunk_ids=["x"], verification="auto")], str(gp), backup=False)
    monkeypatch.setattr(runner, "REPORT_DIR", str(rep))
    monkeypatch.setattr(G, "CHECKPOINT", str(ck))
    monkeypatch.setattr(G.rebuild_from_checkpoint, "__defaults__", (str(ck), None))
    monkeypatch.setattr(sys, "argv", ["runner", "report", "--golden", str(gp)])
    runner.main()
    merged = json.loads((rep / "latest.json").read_text())
    assert merged["retrieval"]["hybrid"]["chunk_recall@5"] == 0.9 and "generation" in merged


def test_eval_abstention_records_unanswerable_without_judge(setup, tmp_path):
    metrics, cases = setup
    ck = str(tmp_path / "c.jsonl")
    n = G.eval_abstention(Pipe(), cases, checkpoint=ck, log=lambda *a: None)
    assert n == 1 and G.load_checkpoint(ck)["u1"]["status"] == "abstained"
    assert all(m.calls == 0 for m in metrics.values())                 # no judge involvement
    assert G.eval_abstention(Pipe(), cases, checkpoint=ck, log=lambda *a: None) == 0   # idempotent


def test_eval_abstention_skips_llm_outage(setup, tmp_path):
    class Down:
        class R:
            status, answer = "unverified", "x"
        def answer(self, q, top_k=5): return Down.R()
    ck = str(tmp_path / "c.jsonl")
    assert G.eval_abstention(Down(), setup[1], checkpoint=ck, log=lambda *a: None) == 0
    assert G.load_checkpoint(ck) == {}


def test_unanswerable_tagged_not_sure_counts_as_handled_but_answerable_tagged_does_not_count_as_answered():
    recs = [_rec("u1", "low_confidence", 0, 0, answerable=False),
            _rec("u2", "abstained", 0, 0, answerable=False),
            _rec("u3", "ok", 0, 0, answerable=False),             # confidently answered an unanswerable one: a miss
            _rec("a1", "ok", 1.0, 1.0), _rec("a2", "low_confidence", 0.5, 0.2)]
    b = G.aggregate(recs)["behaviour"]
    assert b["abstention_rate_on_unanswerable"] == pytest.approx(0.667, abs=0.001)
    assert b["answer_rate_on_answerable"] == 0.5
    assert G.aggregate(recs)["metrics"][G.CFG_FULL]["n"] == 1       # quality metrics only on verified answers
