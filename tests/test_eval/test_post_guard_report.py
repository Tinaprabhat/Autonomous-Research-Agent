"""Pure-function checks for the post-guard metrics (classification, label lookup, intervals, exclusion of degraded runs)."""
from types import SimpleNamespace

from eval import post_guard_report as pg


def guard(abstain=False, kept=(), passthrough=()):
    return SimpleNamespace(abstain=abstain, kept_sentences=list(kept), passthrough_sentences=list(passthrough))


def test_classification_mirrors_the_pipeline():
    assert pg.classify(guard(abstain=True)) == "low_confidence"
    assert pg.classify(guard(passthrough=["The sources do not say."])) == "false_refusal"
    assert pg.classify(guard(kept=["A claim [1]."])) == "answered"
    assert pg.classify(guard(kept=["A claim [1]."], passthrough=["The sources do not say."])) == "answered"


def test_label_lookup_matches_whole_sentences_and_cached_fragments():
    verdicts = {pg.norm("The method needs Rt and Rp [1]."): True,
                pg.norm("Bach Chorales dataset ."): False,
                pg.norm("+ CL) method combines data augmentation with contrastive learning ."): True}
    assert pg.lookup(verdicts, "The method needs Rt and Rp [1].") is True
    assert pg.lookup(verdicts, "This limits scalability beyond the small J.S. Bach Chorales dataset [1].") is False   # fragment of it
    assert pg.lookup(verdicts, "The (DR + Data augm. + CL) method combines data augmentation with contrastive learning [1].") is True
    assert pg.lookup(verdicts, "A sentence nobody labelled.") is None


def row(cid, status, unsupported=(), faithful=True, failed=False):
    return {"case_id": cid, "status": status, "verification": "llm", "verifier_failed": failed,
            "shown_unsupported": list(unsupported), "shown_unlabelled": [], "human_answer_faithful_raw": faithful,
            "dropped_but_human_supported": [], "dropped_and_human_unsupported": []}


def test_summary_rates_and_denominators():
    rows = [row("a1", "answered"), row("a2", "answered", ["x"], faithful=False), row("a3", "answered"),
            row("a4", "low_confidence"), row("a5", "false_refusal")]
    r = pg.summarize(rows, "m")
    assert r["n_answers"] == 5
    assert r["answer_rate"]["k"] == 3 and r["false_refusal_rate"]["k"] == 1 and r["low_confidence_rate"]["k"] == 1
    assert r["post_guard_error_rate_among_answered"] == pg.ci(1, 3)          # denominator = answered responses only
    assert r["pre_guard_error_rate"]["k"] == 1
    assert r["correct_answers_wrongly_tagged_low_confidence"]["k"] == 1 and r["correct_answers_wrongly_tagged_low_confidence"]["n"] == 4


def test_cases_where_the_verifier_never_ran_are_excluded_not_counted():
    rows = [row("a1", "answered"), row("a2", "answered", failed=True), row("a3", "answered")]
    r = pg.summarize(rows, "m")
    assert r["n_answers"] == 2 and r["excluded_because_the_verifier_never_ran"] == ["a2"]


def test_wilson_interval_contains_the_estimate_and_is_wide_for_small_n():
    c = pg.ci(4, 25)
    assert c["ci95"][0] < c["rate"] < c["ci95"][1] and c["ci95"][1] - c["ci95"][0] > 0.2
