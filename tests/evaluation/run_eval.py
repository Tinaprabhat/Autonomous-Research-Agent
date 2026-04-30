"""
Top-level evaluation runner.

Usage (from project root):
    python -m tests.evaluation.run_eval [--retrieval] [--rag] [--all]

Flags:
    --retrieval   Run retrieval-only evaluation (no Ollama needed)
    --rag         Run end-to-end RAG evaluation (Ollama must be running)
    --all         Run both (default if no flag given)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def run_retrieval_eval():
    from backend.app.rag_pipeline.vector_store import VectorStore
    from backend.app.rag_pipeline.hybrid_retriever import HybridRetriever
    from tests.evaluation.eval_retrieval import evaluate_retriever, RETRIEVAL_TEST_CASES

    print("\n═══ RETRIEVAL EVALUATION ════════════════════════════════════════")
    store = VectorStore(dim=384)
    store.load()
    retriever = HybridRetriever(store, store.metadata)
    return evaluate_retriever(retriever, RETRIEVAL_TEST_CASES, k=5)


def run_rag_eval():
    from backend.app.agents.research_agent import ResearchAgent
    from tests.evaluation.eval_rag import evaluate_rag_pipeline, RAG_TEST_CASES

    print("\n═══ RAG END-TO-END EVALUATION ═══════════════════════════════════")
    agent = ResearchAgent()
    return evaluate_rag_pipeline(agent, RAG_TEST_CASES)


def print_summary(report: dict, label: str):
    print(f"\n── {label} aggregate ──")
    for k, v in report["aggregate"].items():
        print(f"  {k:<28}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run RAG system evaluations")
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--rag",       action="store_true")
    parser.add_argument("--all",       action="store_true")
    args = parser.parse_args()

    run_ret = args.retrieval or args.all or not any([args.retrieval, args.rag])
    run_rag = args.rag       or args.all or not any([args.retrieval, args.rag])

    combined = {"timestamp": datetime.now().isoformat()}

    if run_ret:
        ret_report = run_retrieval_eval()
        combined["retrieval"] = ret_report
        print_summary(ret_report, "Retrieval")

    if run_rag:
        rag_report = run_rag_eval()
        combined["rag"] = rag_report
        print_summary(rag_report, "RAG")

    out = "tests/evaluation/full_eval_report.json"
    with open(out, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nFull combined report → {out}\n")


if __name__ == "__main__":
    main()
