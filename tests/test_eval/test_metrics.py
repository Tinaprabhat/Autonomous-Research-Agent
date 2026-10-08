"""Unit tests for eval/metrics.py (the live retrieval metrics: recall@k and MRR)."""
import pytest

from eval.metrics import mean_reciprocal_rank, recall_at_k


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
