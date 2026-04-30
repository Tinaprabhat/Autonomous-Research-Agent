"""
Unit tests for tests/evaluation/metrics.py

Validates every metric function with known inputs and expected outputs.
"""
import pytest
from tests.evaluation.metrics import (
    token_overlap_f1,
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    faithfulness,
    relevance,
    conciseness,
    score_response,
)


# ── token_overlap_f1 ──────────────────────────────────────────────────────────

def test_f1_identical_strings():
    assert token_overlap_f1("hello world", "hello world") == pytest.approx(1.0)


def test_f1_no_overlap():
    assert token_overlap_f1("cat dog", "fish bird") == pytest.approx(0.0)


def test_f1_partial_overlap():
    score = token_overlap_f1("the cat sat", "the dog sat")
    assert 0.0 < score < 1.0


def test_f1_empty_prediction():
    assert token_overlap_f1("", "hello") == pytest.approx(0.0)


def test_f1_empty_reference():
    assert token_overlap_f1("hello", "") == pytest.approx(0.0)


def test_f1_case_insensitive():
    assert token_overlap_f1("Hello World", "hello world") == pytest.approx(1.0)


# ── precision_at_k ────────────────────────────────────────────────────────────

def test_precision_all_relevant():
    retrieved = ["a", "b", "c"]
    assert precision_at_k(retrieved, {"a", "b", "c"}, k=3) == pytest.approx(1.0)


def test_precision_none_relevant():
    retrieved = ["x", "y", "z"]
    assert precision_at_k(retrieved, {"a", "b"}, k=3) == pytest.approx(0.0)


def test_precision_half_relevant():
    retrieved = ["a", "x", "b", "y"]
    assert precision_at_k(retrieved, {"a", "b"}, k=4) == pytest.approx(0.5)


def test_precision_empty_retrieved():
    assert precision_at_k([], {"a"}, k=5) == pytest.approx(0.0)


# ── recall_at_k ───────────────────────────────────────────────────────────────

def test_recall_all_found():
    retrieved = ["a", "b"]
    assert recall_at_k(retrieved, {"a", "b"}, k=2) == pytest.approx(1.0)


def test_recall_none_found():
    retrieved = ["x", "y"]
    assert recall_at_k(retrieved, {"a", "b"}, k=2) == pytest.approx(0.0)


def test_recall_empty_relevant():
    assert recall_at_k(["a"], set(), k=1) == pytest.approx(0.0)


def test_recall_partial():
    retrieved = ["a", "x"]
    assert recall_at_k(retrieved, {"a", "b"}, k=2) == pytest.approx(0.5)


# ── mean_reciprocal_rank ──────────────────────────────────────────────────────

def test_mrr_first_result_relevant():
    assert mean_reciprocal_rank(["a", "b", "c"], {"a"}) == pytest.approx(1.0)


def test_mrr_second_result_relevant():
    assert mean_reciprocal_rank(["x", "a", "b"], {"a"}) == pytest.approx(0.5)


def test_mrr_no_relevant():
    assert mean_reciprocal_rank(["x", "y"], {"a"}) == pytest.approx(0.0)


def test_mrr_empty_retrieved():
    assert mean_reciprocal_rank([], {"a"}) == pytest.approx(0.0)


# ── ndcg_at_k ────────────────────────────────────────────────────────────────

def test_ndcg_perfect_ordering():
    retrieved = ["a", "b"]
    assert ndcg_at_k(retrieved, {"a", "b"}, k=2) == pytest.approx(1.0)


def test_ndcg_no_relevant():
    assert ndcg_at_k(["x", "y"], {"a"}, k=2) == pytest.approx(0.0)


def test_ndcg_between_zero_and_one():
    retrieved = ["x", "a", "b"]
    score = ndcg_at_k(retrieved, {"a", "b"}, k=3)
    assert 0.0 <= score <= 1.0


# ── faithfulness ─────────────────────────────────────────────────────────────

def test_faithfulness_answer_from_context():
    answer  = "Attention allows models to focus on relevant tokens."
    context = ["Attention allows models to focus on relevant tokens in the sequence."]
    score = faithfulness(answer, context)
    assert score > 0.5


def test_faithfulness_no_overlap():
    score = faithfulness("completely unrelated answer", ["different domain text"])
    assert score < 0.5


def test_faithfulness_empty_context():
    assert faithfulness("any answer", []) == pytest.approx(0.0)


# ── relevance ─────────────────────────────────────────────────────────────────

def test_relevance_high_for_keyword_overlap():
    question = "What is self-attention?"
    answer   = "Self-attention is a mechanism in transformer models."
    score = relevance(answer, question)
    assert score > 0.3


def test_relevance_low_for_unrelated():
    question = "What is self-attention?"
    answer   = "The weather in Paris is usually mild."
    score = relevance(answer, question)
    assert score < 0.3


# ── conciseness ───────────────────────────────────────────────────────────────

def test_conciseness_short_answer():
    assert conciseness("Short answer.", max_words=100) == pytest.approx(1.0)


def test_conciseness_exactly_at_limit():
    text = " ".join(["word"] * 100)
    assert conciseness(text, max_words=100) == pytest.approx(1.0)


def test_conciseness_over_limit_decays():
    text = " ".join(["word"] * 150)
    score = conciseness(text, max_words=100)
    assert 0.0 <= score < 1.0


def test_conciseness_double_limit_is_zero():
    text = " ".join(["word"] * 200)
    assert conciseness(text, max_words=100) == pytest.approx(0.0)


# ── score_response ────────────────────────────────────────────────────────────

def test_score_response_returns_dict():
    scores = score_response(
        answer="Attention is a mechanism.",
        question="What is attention?",
        context_chunks=["Attention is used in transformers."],
    )
    assert isinstance(scores, dict)


def test_score_response_has_base_keys():
    scores = score_response(
        answer="Test answer.",
        question="Test question?",
        context_chunks=["context"],
    )
    assert "relevance" in scores
    assert "faithfulness" in scores
    assert "conciseness" in scores


def test_score_response_includes_answer_f1_when_reference_given():
    scores = score_response(
        answer="Test answer.",
        question="Test?",
        context_chunks=["context"],
        reference_answer="Test answer.",
    )
    assert "answer_f1" in scores
    assert scores["answer_f1"] == pytest.approx(1.0)


def test_score_response_includes_latency_score():
    scores = score_response(
        answer="answer",
        question="question",
        context_chunks=["ctx"],
        latency_seconds=2.0,
        latency_budget=10.0,
    )
    assert "latency_score" in scores
    assert scores["latency_score"] == pytest.approx(0.8)


def test_all_scores_in_zero_one_range():
    scores = score_response(
        answer="Attention mechanism in neural networks.",
        question="What is attention?",
        context_chunks=["Attention is used in transformers.", "BERT uses attention."],
        reference_answer="Attention is a core component of transformers.",
        latency_seconds=5.0,
        latency_budget=10.0,
    )
    for key, val in scores.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"
