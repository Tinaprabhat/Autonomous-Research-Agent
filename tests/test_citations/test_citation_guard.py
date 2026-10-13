"""Unit tests for citation_guard.py (NLI-based verification, abstention)."""
from backend.app.citations.guard import _split_sentences, check_citations

CHUNKS = [
    {"text": "BERT pre-trains deep bidirectional transformers using masked language modelling."},
    {"text": "LoRA reduces trainable parameters by injecting low-rank decomposition matrices."},
]


def test_keeps_supported_cited_sentence(fake_verifier):
    r = check_citations("BERT pre-trains bidirectional transformers [1].", CHUNKS, fake_verifier)
    assert r.cited_answer.startswith("BERT") and r.faithfulness_score == 1.0
    assert not r.abstain and r.has_citations


def test_drops_unsupported_cited_sentence(fake_verifier):
    r = check_citations(
        "BERT pre-trains bidirectional transformers [1]. Dolphins sing opera beautifully [2].",
        CHUNKS, fake_verifier)
    assert "Dolphins" not in r.cited_answer
    assert r.dropped_sentences and r.faithfulness_score == 0.5
    assert not r.abstain          # exactly half dropped is not "more than half"


def test_drops_uncited_sentences(fake_verifier):
    r = check_citations("BERT pre-trains bidirectional transformers.", CHUNKS, fake_verifier)
    assert r.cited_answer == "" and r.abstain and not r.has_citations


def test_out_of_range_citation_counts_as_uncited(fake_verifier):
    r = check_citations("BERT pre-trains bidirectional transformers [9].", CHUNKS, fake_verifier)
    assert r.cited_answer == "" and r.abstain


def test_abstains_when_more_than_half_dropped(fake_verifier):
    ans = ("BERT pre-trains bidirectional transformers [1]. Dolphins sing opera [1]. "
           "Penguins drive trucks [2].")
    r = check_citations(ans, CHUNKS, fake_verifier)
    assert r.abstain and r.faithfulness_score < 0.5


def test_no_raw_answer_fallback_when_all_dropped(fake_verifier):
    r = check_citations("Totally unrelated claim here [1].", CHUNKS, fake_verifier)
    assert r.cited_answer == ""          # never silently returns the raw answer


def test_empty_answer_abstains(fake_verifier):
    r = check_citations("   ", CHUNKS, fake_verifier)
    assert r.abstain and r.cited_answer == ""


def test_multi_citation_sentence_uses_best_support(fake_verifier):
    r = check_citations("LoRA reduces trainable parameters [1][2].", CHUNKS, fake_verifier)
    assert r.faithfulness_score == 1.0


def test_marker_after_period_stays_with_its_sentence():
    s = _split_sentences("First claim. [1] Second claim. [2]")
    assert s == ["First claim [1].", "Second claim [2]."]


def test_abbreviations_do_not_split():
    s = _split_sentences("Vaswani et al. introduced attention [1]. It scales well [1].")
    assert len(s) == 2


# ── NLIVerifier windowing (stub model; no download) ──────────────────────────

def test_nli_verifier_scores_best_sentence_window():
    import numpy as np
    from backend.app.citations.guard import NLIVerifier

    class StubModel:
        seen = []
        def predict(self, pairs, batch_size=32):
            StubModel.seen = pairs
            # logits [entailment, neutral, contradiction]; entail only for windows containing "KEY"
            return np.array([[5.0, 0, 0] if "KEY" in p[0] else [0, 5.0, 0] for p in pairs])

    v = NLIVerifier.__new__(NLIVerifier)
    v.model, v._ent_idx = StubModel(), 0
    premise = "One filler sentence. Two filler sentence. Three has the KEY fact. Four filler. Five filler."
    p = v.entailment_batch([(premise, "hyp"), ("short premise without it.", "hyp")])
    assert p[0] > 0.9 and p[1] < 0.1
    assert len(StubModel.seen) > 2                  # long premise was split into windows


def test_only_most_relevant_windows_are_scored():
    import numpy as np
    from backend.app.citations.guard import NLIVerifier
    seen = []

    class StubModel:
        def predict(self, pairs, batch_size=32):
            seen.extend(pairs)
            return np.array([[5.0, 0, 0] if "alpha beta gamma" in p[0].lower() else [0, 5.0, 0] for p in pairs])

    v = NLIVerifier.__new__(NLIVerifier)
    v.model, v._ent_idx = StubModel(), 0
    filler = " ".join(f"Filler sentence number {i} says nothing useful." for i in range(30))
    premise = filler + " Alpha beta gamma method improves recall sharply. " + filler
    p = v.entailment_batch([(premise, "The alpha beta gamma method improves recall.")])
    assert len(seen) == NLIVerifier.MAX_WINDOWS             # not all ~60 windows
    assert p[0] > 0.9                                       # the relevant window was still found


# -- sentence splitting: periods that are not sentence ends (found while hand-labelling, 2026-10-13) --------

def test_splitter_keeps_initials_together():
    from backend.app.citations.guard import _split_sentences
    out = _split_sentences("It needs Rt and Rp [1]. This limits scalability beyond the small J.S. Bach Chorales dataset [1].")
    assert len(out) == 2 and out[1].startswith("This limits") and "J.S. Bach Chorales" in out[1]


def test_splitter_does_not_break_before_a_symbol_or_lowercase():
    from backend.app.citations.guard import _split_sentences
    out = _split_sentences("The (DR + Data augm. + CL) method combines augmentation with contrastive learning [1].")
    assert len(out) == 1


def test_splitter_still_splits_normal_sentences_and_titles_do_not_split():
    from backend.app.citations.guard import _split_sentences
    assert len(_split_sentences("LoRA is small [1]. It is fast [2]. Results improve [1].")) == 3
    assert len(_split_sentences("Dr. Smith showed this [1].")) == 1
