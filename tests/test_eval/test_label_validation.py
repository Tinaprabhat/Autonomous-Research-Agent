"""Statistics + report plumbing for judge / NLI-threshold validation (pure functions, no models)."""
import json

import pytest

from eval import label_validation as lv


def test_wilson_interval_brackets_the_estimate():
    lo, hi = lv.wilson(27, 30)
    assert lo < 0.9 < hi and 0 <= lo and hi <= 1
    assert lv.wilson(0, 0)[0] != lv.wilson(0, 0)[0]            # NaN for n=0


def test_confusion_and_metrics():
    pred = [True, True, False, False, True]
    truth = [True, False, False, True, True]
    tp, fp, fn, tn = lv.confusion(pred, truth)
    assert (tp, fp, fn, tn) == (2, 1, 1, 1)
    m = lv.metrics_from(tp, fp, fn, tn)
    assert m["accuracy"] == 0.6 and m["precision"] == pytest.approx(0.667, abs=1e-3) and m["recall"] == pytest.approx(0.667, abs=1e-3)


def test_cohen_kappa_perfect_chance_and_opposite():
    assert lv.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert lv.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0
    assert lv.cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0


def test_auc_separable_and_tied():
    assert lv.auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert lv.auc([0.5, 0.5], [True, False]) == 0.5
    assert lv.auc([0.1, 0.2], [False, False]) != lv.auc([0.1, 0.2], [False, False])    # NaN: one class only


def test_spearman_monotone():
    assert lv.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert lv.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_analyse_end_to_end(tmp_path, monkeypatch):
    scores = tmp_path / "scores.jsonl"
    rows = [
        {"case_id": "a", "judge_faithfulness": 1.0, "sentences": [
            {"id": "a#0", "nli": 0.95}, {"id": "a#1", "nli": 0.80}]},
        {"case_id": "b", "judge_faithfulness": 0.2, "sentences": [
            {"id": "b#0", "nli": 0.10}, {"id": "b#1", "nli": 0.90}]},
        {"case_id": "c", "judge_faithfulness": 1.0, "sentences": [
            {"id": "c#0", "nli": 0.30}]},                          # judge says faithful, label says not
    ]
    scores.write_text("\n".join(json.dumps(r) for r in rows))
    labels = tmp_path / "labels.jsonl"
    recs = [
        ("sentence", "a#0", True), ("sentence", "a#1", True), ("sentence", "b#0", False),
        ("sentence", "b#1", True), ("sentence", "c#0", False),
        ("answer", "a", True), ("answer", "b", False), ("answer", "c", False)]
    labels.write_text("\n".join(
        json.dumps({"type": t, "id": i, ("supported" if t == "sentence" else "faithful"): v, "labeler": "tester"})
        for t, i, v in recs))
    monkeypatch.setattr(lv, "SCORES", str(scores))

    res = lv.analyse(str(labels))
    nli = res["nli"]
    assert nli["n_sentences"] == 5 and nli["n_supported"] == 3
    assert nli["auc"] == 1.0                                          # supported always outscore unsupported here
    at = nli["at_guard_threshold"]
    assert (at["tp"], at["fp"], at["fn"], at["tn"]) == (3, 0, 0, 2)
    j = res["judge"]
    assert j["n_answers"] == 3 and j["agreement"] == pytest.approx(0.667, abs=1e-3)
    assert j["confusion_judge_pass_vs_label_faithful"] == {"tp": 1, "fp": 1, "fn": 0, "tn": 1}
    assert [d["case_id"] for d in j["disagreements"]] == ["c"]


def test_label_agreement_between_labelers(tmp_path):
    def write(name, vals):
        p = tmp_path / name
        p.write_text("\n".join(json.dumps({"type": "sentence", "id": f"x#{i}", "supported": v, "labeler": name})
                               for i, v in enumerate(vals)))
        return str(p)
    a = write("a.jsonl", [True, True, False, False])
    b = write("b.jsonl", [True, False, False, False])
    r = lv.label_agreement(a, b)
    assert r["sentence"]["n"] == 4 and r["sentence"]["agreement"] == 0.75
