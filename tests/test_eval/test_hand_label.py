"""Tests for eval/hand_label.py (single-keystroke labelling, per-key persistence, resume)."""
import json

import pytest

from eval import hand_label

CASES = [
    {"case_id": f"c{i}", "question": f"Q{i}?", "context": ["ctx a", "ctx b"], "answer": f"A{i}"}
    for i in range(1, 6)
]


@pytest.fixture
def files(tmp_path):
    cases = tmp_path / "answers.jsonl"
    cases.write_text("\n".join(json.dumps(c) for c in CASES))
    return str(cases), str(tmp_path / "labels.jsonl")


def keys(seq):
    it = iter(seq)
    return lambda: next(it)


def labels(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def test_single_keystroke_labels_are_recorded(files):
    cases, out = files
    n = hand_label.run(cases, out, getkey=keys("ynyny"), out=lambda *a, **k: None)
    assert n == 5
    rows = labels(out)
    assert [r["verdict"] for r in rows] == list("ynyny")
    assert [r["case_id"] for r in rows] == [c["case_id"] for c in CASES]
    assert all("timestamp" in r for r in rows)


def test_written_after_every_keystroke_not_batched(files):
    """Simulate a crash on the 3rd prompt: the first two verdicts must already be on disk."""
    cases, out = files
    seq = iter(["y", "n"])

    def getkey():
        try:
            return next(seq)
        except StopIteration:
            assert len(labels(out)) == 2          # persisted before the next prompt
            raise KeyboardInterrupt
    hand_label.run(cases, out, getkey=getkey, out=lambda *a, **k: None)
    assert len(labels(out)) == 2


def test_resume_skips_already_labelled(files):
    cases, out = files
    hand_label.run(cases, out, getkey=keys("yn") , out=lambda *a, **k: None) if False else None
    # label first two, then quit
    hand_label.run(cases, out, getkey=keys(["y", "n", "q"]), out=lambda *a, **k: None)
    assert [r["case_id"] for r in labels(out)] == ["c1", "c2"]
    seen = []
    hand_label.run(cases, out, getkey=keys("yyy"), out=lambda s="", **k: seen.append(s))
    ids = [r["case_id"] for r in labels(out)]
    assert ids == ["c1", "c2", "c3", "c4", "c5"]          # no duplicates, continued where it stopped
    assert any("3 remaining" in str(s) for s in seen)


def test_invalid_keys_are_ignored_until_y_or_n(files):
    cases, out = files
    hand_label.run(cases, out, getkey=keys(["x", "1", " ", "Y", "q"]), out=lambda *a, **k: None)
    assert [r["verdict"] for r in labels(out)] == ["y"]


def test_uppercase_accepted(files):
    cases, out = files
    hand_label.run(cases, out, getkey=keys(["N", "q"]), out=lambda *a, **k: None)
    assert labels(out)[0]["verdict"] == "n"


def test_quit_returns_count_and_stops(files):
    cases, out = files
    assert hand_label.run(cases, out, getkey=keys(["y", "q"]), out=lambda *a, **k: None) == 1


def test_torn_last_line_does_not_break_resume(files):
    cases, out = files
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"case_id": "c1", "verdict": "y", "timestamp": "t"}) + "\n")
        f.write('{"case_id": "c2", "verd')          # simulated crash mid-write
    assert hand_label.load_labelled(out) == {"c1"}


def test_shows_question_context_and_answer(files):
    cases, out = files
    shown = []
    hand_label.run(cases, out, getkey=keys(["q"]), out=lambda s="", **k: shown.append(str(s)))
    text = "\n".join(shown)
    assert "Q1?" in text and "ctx a" in text and "A1" in text and "faithful? (y/n" in text
