"""
Retrieval evaluation: measures how well the hybrid retriever surfaces
the expected chunks for a set of labeled test queries.

Usage (from project root):
    python -m tests.evaluation.eval_retrieval

Expects the vector store to be built (run scripts/build_embeddings.py first).
"""
from __future__ import annotations

import json
import sys
import os

# Allow running from project root without installing as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.evaluation.metrics import precision_at_k, recall_at_k, mean_reciprocal_rank, ndcg_at_k

# ── Labeled test set ──────────────────────────────────────────────────────────
# Each entry: query + set of substrings that should appear in a relevant chunk.
# The evaluator checks whether retrieved chunk texts contain any of those strings.

RETRIEVAL_TEST_CASES: list[dict] = [
    {
        "query": "What is self-attention in transformers?",
        "relevant_keywords": {"attention", "self-attention", "transformer", "query", "key", "value"},
    },
    {
        "query": "How does BERT pre-training work?",
        "relevant_keywords": {"bert", "masked language", "pre-train", "bidirectional"},
    },
    {
        "query": "What is retrieval-augmented generation?",
        "relevant_keywords": {"retrieval", "rag", "generation", "document", "context"},
    },
    {
        "query": "What is FAISS used for?",
        "relevant_keywords": {"faiss", "vector", "similarity", "search", "index"},
    },
    {
        "query": "How does LoRA fine-tuning work?",
        "relevant_keywords": {"lora", "low-rank", "fine-tun", "adapter", "parameter"},
    },
]


def _chunk_is_relevant(chunk_text: str, keywords: set[str]) -> bool:
    text_lower = chunk_text.lower()
    return any(kw in text_lower for kw in keywords)


def evaluate_retriever(retriever, test_cases: list[dict], k: int = 5) -> dict:
    results = []
    for tc in test_cases:
        query = tc["query"]
        keywords = tc["relevant_keywords"]

        retrieved = retriever.search(query, k=k)
        texts = [
            (r["text"] if isinstance(r, dict) else r)
            for r in retrieved
        ]

        relevant_set = {t for t in texts if _chunk_is_relevant(t, keywords)}

        p_k  = precision_at_k(texts, relevant_set, k)
        r_k  = recall_at_k(texts, relevant_set, k)
        mrr  = mean_reciprocal_rank(texts, relevant_set)
        ndcg = ndcg_at_k(texts, relevant_set, k)

        result = {
            "query":       query,
            "precision@k": round(p_k,  4),
            "recall@k":    round(r_k,  4),
            "mrr":         round(mrr,  4),
            "ndcg@k":      round(ndcg, 4),
            "hits":        len(relevant_set),
            "retrieved":   len(texts),
        }
        results.append(result)
        print(f"  [{query[:50]:<50}] P@{k}={p_k:.2f}  R@{k}={r_k:.2f}  MRR={mrr:.2f}  NDCG={ndcg:.2f}")

    avg = {
        "avg_precision@k": round(sum(r["precision@k"] for r in results) / len(results), 4),
        "avg_recall@k":    round(sum(r["recall@k"]    for r in results) / len(results), 4),
        "avg_mrr":         round(sum(r["mrr"]         for r in results) / len(results), 4),
        "avg_ndcg@k":      round(sum(r["ndcg@k"]      for r in results) / len(results), 4),
    }
    return {"per_query": results, "aggregate": avg}


def main():
    from backend.app.rag_pipeline.vector_store import VectorStore
    from backend.app.rag_pipeline.hybrid_retriever import HybridRetriever

    print("Loading vector store …")
    store = VectorStore(dim=384)
    store.load()
    retriever = HybridRetriever(store, store.metadata)
    print(f"Loaded {len(store.metadata)} chunks.\n")

    print(f"Running retrieval evaluation (k=5) on {len(RETRIEVAL_TEST_CASES)} queries …\n")
    report = evaluate_retriever(retriever, RETRIEVAL_TEST_CASES, k=5)

    print("\n── Aggregate scores ──────────────────────────────────────────────")
    for metric, value in report["aggregate"].items():
        print(f"  {metric:<22}: {value:.4f}")

    out_path = "tests/evaluation/retrieval_eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()
