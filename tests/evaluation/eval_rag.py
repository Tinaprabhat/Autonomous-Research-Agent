"""
End-to-end RAG evaluation: runs the full pipeline (retrieve → rerank → generate)
on a set of test questions and scores each answer on relevance, faithfulness,
conciseness, and latency.

Usage (from project root, Ollama must be running):
    python -m tests.evaluation.eval_rag
"""
from __future__ import annotations

import json
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.evaluation.metrics import score_response

# ── Test cases ────────────────────────────────────────────────────────────────
# Optional reference_answer can be None if you just want proxy metrics.

RAG_TEST_CASES: list[dict] = [
    {
        "question": "What is the attention mechanism in transformers?",
        "reference_answer": (
            "The attention mechanism allows transformers to weigh the importance of "
            "different tokens in the sequence relative to each other, using queries, "
            "keys, and values to compute context-aware representations."
        ),
    },
    {
        "question": "How does BERT pre-train on language tasks?",
        "reference_answer": (
            "BERT pre-trains using masked language modelling (MLM) and next sentence "
            "prediction (NSP) on large corpora, learning deep bidirectional representations."
        ),
    },
    {
        "question": "What is retrieval-augmented generation (RAG)?",
        "reference_answer": (
            "RAG combines a retrieval system that fetches relevant documents with a "
            "generative model that synthesises an answer from the retrieved context."
        ),
    },
    {
        "question": "What are the advantages of LoRA fine-tuning?",
        "reference_answer": (
            "LoRA reduces the number of trainable parameters by decomposing weight "
            "updates into low-rank matrices, making fine-tuning memory-efficient and fast."
        ),
    },
    {
        "question": "How does FAISS enable fast similarity search?",
        "reference_answer": (
            "FAISS indexes high-dimensional embeddings and uses approximate nearest-neighbour "
            "algorithms to find similar vectors at scale without exhaustive pairwise comparison."
        ),
    },
]


def evaluate_rag_pipeline(agent, test_cases: list[dict], latency_budget: float = 15.0) -> dict:
    results = []
    for tc in test_cases:
        question  = tc["question"]
        reference = tc.get("reference_answer")

        t0 = time.perf_counter()
        answer = agent.run(question)
        elapsed = time.perf_counter() - t0

        # Collect context used: re-retrieve for scoring (agent doesn't expose it)
        context_chunks: list[str] = []
        if hasattr(agent, "retriever"):
            retrieved = agent.retriever.search(question, k=5)
            context_chunks = [
                (r["text"] if isinstance(r, dict) else r)
                for r in retrieved
            ]

        scores = score_response(
            answer=answer,
            question=question,
            context_chunks=context_chunks,
            reference_answer=reference,
            latency_seconds=elapsed,
            latency_budget=latency_budget,
        )

        result = {
            "question": question,
            "answer":   answer[:300],
            "latency_s": round(elapsed, 2),
            **{k: round(v, 4) for k, v in scores.items()},
        }
        results.append(result)

        print(f"\n  Q: {question}")
        print(f"  A: {answer[:120]}…")
        print(f"  Latency: {elapsed:.1f}s  " +
              "  ".join(f"{k}={v:.2f}" for k, v in scores.items()))

    # Aggregate over all numeric score fields
    score_keys = [k for k in results[0] if k not in ("question", "answer", "latency_s")]
    agg = {
        f"avg_{k}": round(sum(r[k] for r in results) / len(results), 4)
        for k in score_keys
    }
    return {"per_question": results, "aggregate": agg}


def main():
    from backend.app.agents.research_agent import ResearchAgent

    print("Initialising Research Agent (loads FAISS + models) …")
    agent = ResearchAgent()
    print("Ready.\n")

    print(f"Running RAG evaluation on {len(RAG_TEST_CASES)} questions …")
    report = evaluate_rag_pipeline(agent, RAG_TEST_CASES)

    print("\n── Aggregate scores ──────────────────────────────────────────────")
    for metric, value in report["aggregate"].items():
        print(f"  {metric:<25}: {value:.4f}")

    out_path = "tests/evaluation/rag_eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()
