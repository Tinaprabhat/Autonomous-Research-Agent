"""Pure parts of eval/answer_quality.py: judgment parsing, intervals, AUROC, label calibration, judge sanity gate."""
import pytest

from eval import answer_quality as aq


def test_parse_judgment_computes_completeness_and_relevance():
    j = aq.parse_judgment('{"key_facts": [{"fact": "a", "covered": true}, {"fact": "b", "covered": false}], '
                          '"sentences": [{"id": 1, "relevant": true}, {"id": 2, "relevant": true}, {"id": 3, "relevant": false}]}')
    assert j["completeness"] == 0.5 and j["n_facts"] == 2
    assert j["relevance"] == pytest.approx(2 / 3) and j["n_sentences"] == 3


def test_parse_judgment_rejects_garbage_and_empty_fact_lists():
    assert aq.parse_judgment("sure, looks good") is None
    assert aq.parse_judgment('{"key_facts": [], "sentences": []}') is None
    assert aq.parse_judgment('{"sentences": []}') is None
    assert aq.parse_judgment("") is None


def test_parse_judgment_handles_text_around_the_json_and_string_booleans():
    j = aq.parse_judgment('Here you go: {"key_facts": [{"fact": "a", "covered": "true"}], "sentences": []} done')
    assert j["completeness"] == 1.0 and j["relevance"] is None


def test_bootstrap_mean_is_deterministic_and_brackets_the_mean():
    v = [0.0, 0.5, 1.0, 1.0, 0.5, 0.0, 1.0]
    a, b = aq.bootstrap_mean(v), aq.bootstrap_mean(v)
    assert a == b and a["ci95"][0] <= a["mean"] <= a["ci95"][1] and a["n"] == 7
    assert aq.bootstrap_mean([])["mean"] is None


def test_auroc_cases():
    assert aq.auroc([1, 2], [0, 0.5]) == 1.0
    assert aq.auroc([0, 0.5], [1, 2]) == 0.0
    assert aq.auroc([1, 1], [1, 1]) == 0.5
    assert aq.auroc([], [1]) is None


def row(status, score, unsupported=()):
    return {"status": status, "guard_score": score, "shown_unsupported": list(unsupported)}


def test_label_calibration_reports_reliability_per_label():
    rows = [row("answered", 1.0), row("answered", 1.0, ["x"]), row("answered", 0.5), row("low_confidence", 0.0),
            row("low_confidence", 0.0), row("false_refusal", 1.0)]
    c = aq.calibration(rows)
    by = c["fraction_of_responses_with_no_unsupported_claim_by_label"]
    assert by["answered"]["k"] == 2 and by["answered"]["n"] == 3
    assert by["low_confidence"]["k"] == 2 and by["false_refusal"]["rate"] == 1.0
    assert c["n_responses_with_an_unsupported_claim"] == 1
    assert c["flagged_low_confidence_among_responses_with_error"]["k"] == 0


def test_a_score_that_is_higher_for_errors_gets_an_auroc_below_one_half():
    rows = [row("answered", 1.0, ["x"]), row("answered", 1.0, ["x"]), row("answered", 0.5), row("answered", 0.0)]
    assert aq.calibration(rows)["guard_score_predicts_correctness"]["auroc"] < 0.5


def test_the_judge_sanity_gate_requires_controls_to_separate():
    def j(c, r):
        return {"completeness": c, "relevance": r}
    rows = [{"case_id": f"c{i}"} for i in range(4)]
    good = {}
    for i in range(4):
        good[("real", f"c{i}")] = j(0.6, 0.7)
        good[("reference", f"c{i}")] = j(1.0, 1.0)
        good[("shuffled", f"c{i}")] = j(0.0, 0.0)
    assert aq.summarise_judge(good, rows)["judge_discriminates"] is True
    bad = dict(good)
    for i in range(4):
        bad[("shuffled", f"c{i}")] = j(0.8, 0.9)                 # a judge that gives the wrong answers credit
    assert aq.summarise_judge(bad, rows)["judge_discriminates"] is False
    unusable = {("real", "c0"): {"unusable": True}}
    assert aq.summarise_judge(unusable, rows)["real"]["n_unusable"] == 1
