"""
Core evaluation metrics for the RAG pipeline.

All metrics return floats in [0, 1] (higher = better) unless noted.
No external scoring APIs — everything runs locally.
"""
from __future__ import annotations

import re
import time
from typing import Callable


# ── Token overlap helpers ─────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def token_overlap_f1(prediction: str, reference: str) -> float:
    """Unigram F1 between prediction and reference (alias for ROUGE-1 F1)."""
    pred_tokens = _tokenize(prediction)
    ref_tokens  = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved chunks that are relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for t in top_k if t in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant chunks found in top-k results."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for t in top_k if t in relevant)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """MRR: 1/rank of the first relevant result (0 if none found)."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalised DCG at k (binary relevance: 1 if relevant, 0 otherwise)."""
    import math
    dcg  = sum(1 / math.log2(i + 2) for i, t in enumerate(retrieved[:k]) if t in relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ── Answer quality metrics ────────────────────────────────────────────────────

def faithfulness(answer: str, context_chunks: list[str]) -> float:
    """
    Proxy faithfulness: max token-overlap F1 between answer and any context chunk.
    High faithfulness → answer text mostly comes from retrieved context.
    """
    if not context_chunks:
        return 0.0
    return max(token_overlap_f1(answer, chunk) for chunk in context_chunks)


def relevance(answer: str, question: str) -> float:
    """Proxy relevance: token-overlap F1 between answer and question keywords."""
    return token_overlap_f1(answer, question)


def conciseness(answer: str, max_words: int = 100) -> float:
    """
    Penalises answers that exceed max_words.
    Returns 1.0 if within limit, decays linearly beyond.
    """
    n_words = len(answer.split())
    if n_words <= max_words:
        return 1.0
    return max(0.0, 1.0 - (n_words - max_words) / max_words)


# ── Latency metric ────────────────────────────────────────────────────────────

def measure_latency(fn: Callable, *args, **kwargs) -> tuple[float, object]:
    """
    Times `fn(*args, **kwargs)` and returns (elapsed_seconds, result).
    """
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    return elapsed, result


# ── Aggregate scorer ──────────────────────────────────────────────────────────

def score_response(
    *,
    answer: str,
    question: str,
    context_chunks: list[str],
    reference_answer: str | None = None,
    latency_seconds: float | None = None,
    latency_budget: float = 10.0,
) -> dict[str, float]:
    """
    Return a dict of all applicable metrics for one QA pair.
    """
    scores: dict[str, float] = {
        "relevance":    relevance(answer, question),
        "faithfulness": faithfulness(answer, context_chunks),
        "conciseness":  conciseness(answer),
    }
    if reference_answer:
        scores["answer_f1"] = token_overlap_f1(answer, reference_answer)
    if latency_seconds is not None:
        scores["latency_score"] = max(0.0, 1.0 - latency_seconds / latency_budget)
    return scores
