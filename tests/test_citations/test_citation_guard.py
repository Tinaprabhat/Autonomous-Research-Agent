"""Unit tests for guard.py: layer A screen, layer B entailment chain with fallbacks, hedge passthrough, abstention."""
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


# -- layered guard: layer A (overlap + specifics), layer B (entailment chain), hedge passthrough ---------------

import pytest

from backend.app.citations.guard import (EntailmentChain, anchors, check_citations, is_hedge,
                                          lexical_overlap, missing_anchors)
from backend.app.citations.llm_verifier import LLMEntailment, build_prompt, parse_results


class Scripted:
    """Entailment verifier returning fixed flags (or None = cannot decide); records what it was asked."""
    def __init__(self, flags=True, name="scripted"):
        self.flags, self.name, self.seen = flags, name, []

    def verify(self, items):
        self.seen.append(items)
        if self.flags is None:
            return None
        return [self.flags] * len(items) if isinstance(self.flags, bool) else list(self.flags)


def chain(verifier=None, mode="llm"):
    return EntailmentChain(llm=verifier, mode=mode)


# -- layer A -----------------------------------------------------------------------------------------------

def test_lexical_overlap_is_the_share_of_sentence_words_found_in_the_chunk():
    assert lexical_overlap("LoRA reduces trainable parameters", "LoRA reduces trainable parameters by injecting") == 1.0
    assert lexical_overlap("LoRA reduces cost", "LoRA reduces trainable parameters") == pytest.approx(2 / 3)
    assert lexical_overlap("", "anything") == 0.0


def test_anchors_and_missing_anchors():
    a = anchors("CARROT uses only 899 online tokens with RAPTOR.")
    assert "899" in a and "raptor" in a
    cover, miss = missing_anchors("It uses 12 tokens with RAPTOR.", ["carrot needs 899 tokens and raptor is a baseline"])
    assert cover == 0.5 and miss == ["12"]
    assert missing_anchors("nothing special here", ["x"]) == (1.0, [])


def test_wrong_chunk_citation_is_dropped_by_the_overlap_floor_before_any_model_runs():
    v = Scripted(True)
    r = check_citations("Dolphins sing opera on distant planets [2].", CHUNKS, chain(v))
    assert r.dropped_sentences and r.details[0]["reason"] == "low_overlap"
    assert v.seen == []                                            # layer B was never asked


def test_a_changed_number_is_dropped_by_the_specifics_gate():
    v = Scripted(True)
    r = check_citations("BERT pre-trains 12 deep bidirectional transformers [1].", CHUNKS, chain(v))
    assert r.dropped_sentences and r.details[0]["reason"] == "missing_specifics" and r.details[0]["missing"] == ["12"]
    assert v.seen == []


def test_specifics_may_come_from_any_cited_chunk():
    ans = "BERT pre-trains transformers and LoRA reduces trainable parameters [1][2]."
    assert check_citations(ans, CHUNKS, chain(Scripted(True))).kept_sentences


def test_thresholds_are_overridable():
    s = "LoRA reduces trainable parameters and removes the need for extra hardware [2]."
    assert check_citations(s, CHUNKS, chain(Scripted(True)), overlap_threshold=0.3).kept_sentences
    assert check_citations(s, CHUNKS, chain(Scripted(True)), overlap_threshold=0.9).dropped_sentences


# -- layer B: the entailment chain ---------------------------------------------------------------------------------

def test_layer_b_only_sees_sentences_that_passed_layer_a():
    v = Scripted(True)
    r = check_citations("LoRA reduces trainable parameters [2]. Dolphins sing opera on planets [2].", CHUNKS, chain(v))
    assert len(v.seen) == 1 and len(v.seen[0]) == 1 and v.seen[0][0][1].startswith("LoRA reduces")
    assert r.kept_sentences and r.dropped_sentences


def test_layer_b_receives_all_cited_chunks_for_the_claim():
    v = Scripted(True)
    check_citations("BERT pre-trains transformers and LoRA reduces trainable parameters [1][2].", CHUNKS, chain(v))
    texts, claim = v.seen[0][0]
    assert len(texts) == 2 and "BERT" in texts[0] and "LoRA" in texts[1]


def test_not_entailed_sentence_is_dropped_with_the_verifier_named():
    r = check_citations("LoRA reduces trainable parameters [2].", CHUNKS, chain(Scripted(False, "llm")))
    assert r.dropped_sentences and r.details[0]["reason"] == "not_entailed" and r.details[0]["verifier"] == "llm"
    assert r.verification == "llm"


def test_when_the_llm_verifier_cannot_decide_the_result_is_lexical_only_not_verified():
    llm = Scripted(None, "llm")
    r = check_citations("LoRA reduces trainable parameters [2].", CHUNKS, chain(llm))
    assert r.kept_sentences and r.verification == "lexical_only" and llm.seen          # asked once, no second verifier


def test_when_the_llm_verifier_raises_the_answer_survives_as_lexical_only():
    class Boom:
        name = "llm"
        def verify(self, items):
            raise RuntimeError("provider down")
    r = check_citations("LoRA reduces trainable parameters [2].", CHUNKS, chain(Boom()))
    assert r.kept_sentences and r.verification == "lexical_only"


def test_no_verifier_at_all_degrades_to_lexical_only_not_to_verified():
    assert check_citations("LoRA reduces trainable parameters [2].", CHUNKS, chain(None)).verification == "lexical_only"
    assert check_citations("LoRA reduces trainable parameters [2].", CHUNKS, None).verification == "lexical_only"


def test_mode_off_runs_layer_a_only_and_never_calls_the_verifier():
    llm = Scripted(True, "llm")
    off = check_citations("LoRA reduces trainable parameters [2].", CHUNKS, chain(llm, mode="off"))
    assert off.verification == "lexical_only" and not llm.seen


def test_an_unsupported_mode_such_as_the_removed_nli_falls_back_to_llm():
    assert chain(Scripted(True), mode="nli").mode == "llm"


def test_a_wrong_length_reply_is_treated_as_unable_to_decide():
    class Short:
        name = "llm"
        def verify(self, items):
            return [True]                                                    # but 2 sentences were asked about
    ans = "LoRA reduces trainable parameters [2]. BERT pre-trains deep transformers [1]."
    assert check_citations(ans, CHUNKS, chain(Short())).verification == "lexical_only"


# -- hedge passthrough and abstention -------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "The sources do not contain this information.",
    "I couldn't find any mention of that dataset.",
    "Not mentioned in the provided passages.",
    "I'm not sure about this.",
    "There is no information about it in the sources.",
    "I’m sorry, but none of the provided sources contain information about Australia’s capital.",
    "None of the retrieved passages mention that dataset.",
    "The sources provided do not cover this topic."])
def test_hedge_phrases_are_recognised(text):
    assert is_hedge(text)


def test_ordinary_claims_and_long_sentences_are_not_hedges():
    assert not is_hedge("LoRA reduces trainable parameters by injecting low-rank matrices.")
    assert not is_hedge("The authors state that " + "x" * 300 + " but the sources do not mention more.")


def test_uncited_hedge_passes_through_and_is_excluded_from_the_score():
    ans = "LoRA reduces trainable parameters [2]. The sources do not mention training cost."
    r = check_citations(ans, CHUNKS, chain(Scripted(True)))
    assert r.passthrough_sentences == ["The sources do not mention training cost."]
    assert r.faithfulness_score == 1.0 and not r.abstain
    assert "do not mention" in r.cited_answer and r.cited_answer.startswith("LoRA")


def test_hedge_only_answer_is_not_abstained_and_never_calls_a_verifier():
    v = Scripted(True)
    r = check_citations("I cannot find any mention of that in the provided sources.", CHUNKS, chain(v))
    assert not r.abstain and r.cited_answer and r.faithfulness_score == 1.0 and not r.has_citations and not v.seen


def test_cited_hedge_sentence_is_still_checked():
    r = check_citations("The passage does not mention dolphins singing opera on planets [1].", CHUNKS, chain(Scripted(True)))
    assert not r.passthrough_sentences


def test_uncited_non_hedge_is_still_dropped():
    r = check_citations("BERT pre-trains transformers.", CHUNKS, chain(Scripted(True)))
    assert r.dropped_sentences and not r.passthrough_sentences


def test_abstain_counts_only_checkable_sentences():
    ans = "Cats sing [1]. Dogs fly [1]. The sources do not mention more. No further information is given."
    assert check_citations(ans, CHUNKS, chain(Scripted(True))).abstain


# -- the LLM verifier: prompt and strict parsing ----------------------------------------------------------------

def test_prompt_numbers_every_claim_and_joins_multiple_passages():
    p = build_prompt([("passage one", "claim one"), (["a", "b"], "claim two")])
    assert "[1] PASSAGE: passage one" in p and "CLAIM: claim one" in p and "[2] PASSAGE: a\n---\nb" in p


def test_parse_accepts_well_formed_json_and_rejects_everything_else():
    assert parse_results('{"results": [{"id": 1, "supported": true}, {"id": 2, "supported": false}]}', 2) == [True, False]
    assert parse_results('noise {"results":[{"id":1,"supported":false}]} noise', 1) == [False]
    assert parse_results('{"results": [{"id": 1, "supported": true}]}', 2) is None           # missing a claim
    assert parse_results('{"results": [{"id": 1, "supported": true}, {"id": 5, "supported": true}]}', 2) is None
    assert parse_results("I think they are fine.", 2) is None
    assert parse_results("", 1) is None


def test_llm_entailment_returns_none_on_provider_error_or_garbage():
    class Down:
        def generate(self, prompt):
            raise RuntimeError("429")

    class Garbage:
        def generate(self, prompt):
            return "sure, looks fine!"

    assert LLMEntailment(Down()).verify([("p", "c")]) is None
    assert LLMEntailment(Garbage()).verify([("p", "c")]) is None


def test_llm_entailment_makes_one_call_per_batch_and_splits_large_answers():
    calls = []

    class Good:
        def generate(self, prompt):
            calls.append(prompt)
            n = prompt.count("CLAIM:")
            return '{"results": [' + ",".join(f'{{"id": {i}, "supported": true}}' for i in range(1, n + 1)) + "]}"

    assert LLMEntailment(Good()).verify([("p", "c")] * 3) == [True] * 3 and len(calls) == 1
    calls.clear()
    assert len(LLMEntailment(Good()).verify([("p", "c")] * 30)) == 30 and len(calls) == 3      # 12 + 12 + 6


def test_llm_entailment_asks_for_a_large_output_budget_when_the_llm_supports_it():
    seen = {}

    class Budgeted:
        def generate(self, prompt, use_cache=True, max_tokens=None):
            seen["max_tokens"] = max_tokens
            return '{"results": [{"id": 1, "supported": true}]}'

    class Plain:
        def generate(self, prompt):
            return '{"results": [{"id": 1, "supported": false}]}'

    assert LLMEntailment(Budgeted()).verify([("p", "c")]) == [True] and seen["max_tokens"] >= 1000
    assert LLMEntailment(Plain()).verify([("p", "c")]) == [False]               # fakes without the parameter still work


class _RateLimited(Exception):
    def __init__(self, retry_after, daily=False):
        super().__init__("429")
        self.retry_after, self.daily = retry_after, daily


def test_llm_entailment_waits_out_a_short_per_minute_limit_once(monkeypatch):
    slept = []
    monkeypatch.setattr("backend.app.citations.llm_verifier.time.sleep", lambda s: slept.append(s))

    class Flaky:
        n = 0
        def generate(self, prompt):
            Flaky.n += 1
            if Flaky.n == 1:
                raise _RateLimited(5.0)
            return '{"results": [{"id": 1, "supported": true}]}'

    assert LLMEntailment(Flaky()).verify([("p", "c")]) == [True] and slept and 5.0 <= slept[0] < 7


def test_llm_entailment_does_not_wait_for_a_daily_limit_or_a_long_wait(monkeypatch):
    slept = []
    monkeypatch.setattr("backend.app.citations.llm_verifier.time.sleep", lambda s: slept.append(s))

    class Daily:
        def generate(self, prompt):
            raise _RateLimited(5.0, daily=True)

    class Long:
        def generate(self, prompt):
            raise _RateLimited(300.0)

    assert LLMEntailment(Daily()).verify([("p", "c")]) is None
    assert LLMEntailment(Long()).verify([("p", "c")]) is None
    assert slept == []


def test_llm_entailment_gives_up_after_one_retry(monkeypatch):
    monkeypatch.setattr("backend.app.citations.llm_verifier.time.sleep", lambda s: None)

    class Always:
        calls = 0
        def generate(self, prompt):
            Always.calls += 1
            raise _RateLimited(1.0)

    assert LLMEntailment(Always()).verify([("p", "c")]) is None and Always.calls == 2
