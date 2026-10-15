"""Mutation generators and helpers behind the guard stress test (pure functions, no models)."""
import pytest

from eval import guard_stress as gs
from backend.app.citations.guard import anchors
from eval.guard_stress_scoring import specifics_coverage


def test_number_swap_changes_the_number_and_keeps_the_rest():
    out = gs._swap_number("CARROT uses only 899 online input tokens, whereas HYDE consumes 8192 tokens.")
    assert out and "899" not in out and "8192" in out and out.endswith("tokens.")
    assert gs._swap_number("No digits here at all.") is None


def test_antonym_flip_reverses_meaning_words_only():
    assert gs._flip_antonym("Accuracy increases with more steps.") == "Accuracy decreases with more steps."
    assert gs._flip_antonym("Nothing to flip here") is None


def test_negation_reuses_every_original_word():
    s = "LoRA reduces trainable parameters."
    out = gs._negate(s)
    assert out == "It is not the case that LoRA reduces trainable parameters."          # mixed-case names keep their casing
    assert gs._negate("Accuracy improves.") == "It is not the case that accuracy improves."


def test_qualifier_and_conflation_append_foreign_content():
    s = "BERT pre-trains transformers."
    assert gs._qualify(s, 0).startswith("BERT pre-trains transformers,") and len(gs._qualify(s, 0)) > len(s) + 20
    c = gs._conflate(s, "CARROT uses 899 tokens.")
    assert "Furthermore, CARROT uses 899 tokens." in c            # names and acronyms keep their casing
    assert "Furthermore, the attack" in gs._conflate(s, "The attack may modify samples.")


def test_build_is_deterministic_and_labels_hold_by_construction(tmp_path):
    a, b = gs.build(), gs.build()
    assert a == b and len(a) > 150
    cats = {r["category"] for r in a}
    assert {"real_supported", "real_unsupported", "negation", "qualifier", "conflation", "wrong_chunk"} <= cats
    for r in a:
        assert r["label"] == ("supported" if r["category"] == "real_supported" else "unsupported")
    # a synthetic item is never identical to the supported sentence it came from (except wrong_chunk: the chunk changes)
    real = {r["sentence"] for r in a if r["category"] == "real_supported"}
    assert all(r["sentence"] not in real for r in a if r["category"] in ("number_swap", "negation", "qualifier", "conflation"))


def test_specifics_anchors_and_coverage():
    s = "CARROT uses only 899 online tokens with RAPTOR."
    a = anchors(s)
    assert "899" in a and "raptor" in a and "carrot" in a            # numbers, acronyms and names are the checkable specifics
    chunk = "carrot needs 899 tokens and raptor is a baseline"
    assert specifics_coverage(s, chunk) == 1.0
    assert specifics_coverage("It uses 12 tokens with RAPTOR.", chunk) == 0.5
    assert specifics_coverage("nothing special here", chunk) == 1.0   # no anchors: nothing to contradict
