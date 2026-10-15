"""
runner.py — ablation + generation-quality evaluation against the golden set.

    python -m eval.runner retrieval [--k 5] [--local]              # no LLM: recall@k / MRR per config
    python -m eval.runner generation [--limit 40] [--local]        # RAGAS metrics, with vs without guard
    python -m eval.runner all --limit 40 --dump-answers            # both + answers file for hand_label.py

Configs (ablation table): dense / bm25 / hybrid / hybrid+rerank / +guard (full pipeline).
Retrieval metrics (recall@k, MRR) are non-LLM and reuse eval/metrics.py. Generation metrics
use RAGAS with a Groq judge (eval/ragas_metrics.py). Answerable-vs-unanswerable behaviour is scored
separately (abstention rate on unanswerable, answer rate on answerable).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
# [DEAD CODE] unused import (checked 2026-10-15)
# import time
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval import golden
from eval.metrics import mean_reciprocal_rank, recall_at_k

RETRIEVAL_CONFIGS = ["dense", "bm25", "hybrid", "hybrid+rerank"]
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")


# ── retrieval ablation ───────────────────────────────────────────────────────

def retrieve_for_config(pipeline, config: str, question: str, k: int) -> list[dict]:
    r = pipeline.retriever
    pool = max(k * 4, 20)
    if config == "dense":
        return r.dense_search(question, k)
    if config == "bm25":
        return r.bm25_search(question, k)
    if config == "hybrid":
        return r.search(question, k=k, rewrite=False)
    if config == "hybrid+rerank":
        cands = r.search(question, k=pool, rewrite=False)
        return pipeline.reranker.rerank(question, cands, top_k=k)
    raise ValueError(config)


def eval_retrieval(pipeline, cases, k: int = 5) -> dict:
    """recall@k / MRR (chunk-level and paper-level) per config, on answerable cases."""
    answerable = [c for c in cases if c.answerable]
    out = {}
    for cfg in RETRIEVAL_CONFIGS:
        # [DEAD CODE] `prec_p` was never filled or read (checked 2026-10-08); removed from the unpacking.
        rec, mrr, rec_p = [], [], []
        for c in answerable:
            docs = retrieve_for_config(pipeline, cfg, c.question, k)
            ids = [d["chunk_id"] for d in docs]
            papers = list(dict.fromkeys(d.get("arxiv_id") for d in docs))
            if c.gold_chunk_ids:
                gold = set(c.gold_chunk_ids)
                rec.append(recall_at_k(ids, gold, k))
                mrr.append(mean_reciprocal_rank(ids, gold))
            if c.gold_arxiv_ids:
                gp = set(c.gold_arxiv_ids)
                rec_p.append(recall_at_k(papers, gp, k))
        out[cfg] = {
            f"chunk_recall@{k}": _mean(rec), "chunk_mrr": _mean(mrr),
            f"paper_recall@{k}": _mean(rec_p), "n": len(answerable),
        }
    return out


# ── generation metrics (RAGAS, see eval/ragas_metrics.py) ─────────────────────────────────────────────────

def ragas_metrics(judge):
    """{name: metric} for a RagasJudge (faithfulness, hallucination, correctness, completeness)."""
    return judge.metrics()


def raw_answer(pipeline, question: str, chunks: list[dict]) -> str:
    """The same prompt + LLM as the pipeline, but WITHOUT the citation guard."""
    from backend.app.pipeline import RAG_PROMPT
    from backend.app.citations.formatter import format_context_with_citations
    ctx, _ = format_context_with_citations(chunks)
    return pipeline.llm.generate(RAG_PROMPT.format(context=ctx[:6000], question=question))


def eval_generation(*args, **kwargs):
    """Resumable RAGAS generation eval (implemented in eval/generation.py)."""
    from eval import generation
    return generation.eval_generation(*args, **kwargs)


def composite(generation: dict, config: str = "+guard (full)") -> float | None:
    """Mean of faithfulness / correctness / completeness — the CI gate value."""
    m = generation["metrics"].get(config, {})
    vals = [m[k] for k in ("faithfulness", "correctness", "completeness") if m.get(k) is not None]
    return round(statistics.mean(vals), 3) if vals else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


# ── CLI ──────────────────────────────────────────────────────────────────────

def _save(report: dict, ts: str) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    for name in (f"report_{ts}.json", "latest.json"):
        with open(os.path.join(REPORT_DIR, name), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    print("report ->", os.path.join(REPORT_DIR, "latest.json"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["retrieval", "generation", "all", "report", "abstention"])
    ap.add_argument("--golden", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=40, help="max cases for LLM-judged generation metrics")
    ap.add_argument("--metrics", nargs="*", default=None)
    ap.add_argument("--include-unreviewed", action="store_true")
    ap.add_argument("--dump-answers", action="store_true", help="write answers file for eval/hand_label.py")
    ap.add_argument("--local", action="store_true", help="force embedded local Qdrant")
    a = ap.parse_args()
    if a.local:
        os.environ["QDRANT_FORCE_LOCAL"] = "1"

    cases = golden.load(a.golden, only_reviewed=not a.include_unreviewed)
    if not cases:
        sys.exit("No reviewed cases in the golden set. Review eval/data/golden_set.json "
                 "(set \"reviewed\": true) or pass --include-unreviewed.")
    if a.mode == "abstention":          # unanswerable cases only: pipeline calls, zero judge tokens
        from backend.app.services import build_pipeline
        from eval import generation
        pipeline, _ = build_pipeline(use_rewriter=False)
        print("newly run:", generation.eval_abstention(pipeline, cases, a.k))
        a.mode = "report"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    latest = os.path.join(REPORT_DIR, "latest.json")
    report = {}
    if os.path.exists(latest):          # merge: running one mode must not erase the other's results
        try:
            with open(latest, encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError):
            report = {}
    report.update({"timestamp": ts, "k": a.k, "n_cases": len(cases),
                   "n_answerable": sum(c.answerable for c in cases),
                   "n_unanswerable": sum(not c.answerable for c in cases)})

    if a.mode == "report":              # re-aggregate from the checkpoint; no pipeline, no judge calls
        from eval import generation
        report["generation"] = generation.rebuild_from_checkpoint(golden_total=len(cases))
        report["composite"] = composite(report["generation"])
        print(json.dumps({k: report["generation"][k] for k in ("metrics", "behaviour")}, indent=2))
        _save(report, ts)
        return

    from backend.app.services import build_pipeline
    pipeline, _ = build_pipeline(use_rewriter=False)
    if a.mode in ("retrieval", "all"):
        report["retrieval"] = eval_retrieval(pipeline, cases, a.k)
        print(json.dumps(report["retrieval"], indent=2))
    if a.mode in ("generation", "all"):
        dump = None
        if a.dump_answers:
            os.makedirs("data/eval_runs", exist_ok=True)
            dump = "data/eval_runs/answers_latest.jsonl"
        report["generation"] = eval_generation(pipeline, cases, a.k, a.limit, a.metrics, dump_answers=dump)
        report["composite"] = composite(report["generation"])
        # [DEAD CODE] unused local `stopped_early` (checked 2026-10-08); the value is read from the report below.
        # stopped_early = report["generation"].get("stopped_early")
        print(json.dumps({k: report["generation"][k] for k in ("metrics", "behaviour")}, indent=2))
        print("composite:", report["composite"])
        if dump:
            print("answers ->", dump)

    _save(report, ts)
    if a.mode in ("generation", "all") and report["generation"].get("stopped_early"):
        print("PAUSED:", report["generation"]["stopped_early"])
        print("Re-run the same command later to resume from data/eval_runs/generation_ckpt.jsonl.")
        sys.exit(3)


if __name__ == "__main__":
    main()
