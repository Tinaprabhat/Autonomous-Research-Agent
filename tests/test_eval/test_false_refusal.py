"""False-refusal tracking: a hedge-only answer on an answerable question must not count as answered."""
import json

from backend.app.telemetry import observability
from eval.generation import aggregate


def rec(case, status="ok", hedged=False, answerable=True):
    return {"case_id": case, "answerable": answerable, "status": status, "hedged": hedged, "scores": {}}


def test_hedged_answers_count_as_false_refusals_not_answers():
    b = aggregate([rec("a1"), rec("a2"), rec("a3", hedged=True), rec("a4", status="low_confidence")])["behaviour"]
    assert b["answerable_total"] == 4 and b["answerable_answered"] == 2
    assert b["answerable_hedged"] == 1 and b["false_refusal_rate_on_answerable"] == 0.25
    assert b["answer_rate_on_answerable"] == 0.5


def test_old_checkpoint_records_without_the_field_still_aggregate():
    r = {"case_id": "a1", "answerable": True, "status": "ok", "scores": {}}      # written before 'hedged' existed
    assert aggregate([r])["behaviour"]["answerable_answered"] == 1


def test_metrics_summary_reports_the_hedged_rate(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in (
        {"status": "ok", "latency_ms": 10, "hedged": False}, {"status": "ok", "latency_ms": 20, "hedged": True},
        {"status": "ok", "latency_ms": 30}, {"status": "low_confidence", "latency_ms": 40, "hedged": False})))
    s = observability.summarize(str(p))
    assert s["hedged_answers"] == 1 and s["hedged_rate"] == 0.25
