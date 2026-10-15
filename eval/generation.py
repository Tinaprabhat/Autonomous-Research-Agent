"""
generation.py — resumable RAGAS generation-quality evaluation.

For each answerable golden case the same retrieved context is answered twice:
  * "hybrid+rerank (no guard)": raw LLM answer, no citation verification
  * "+guard (full)":            the production pipeline (citation guard, abstention, output guard)
and scored with RAGAS metrics (faithfulness, hallucination, correctness, completeness; Groq judge, see eval/ragas_metrics.py).
Unanswerable cases only measure abstention behaviour.

Every finished case is appended (fsync'd) to a checkpoint JSONL, so a crash, Ctrl-C, or an exhausted
free-tier DAILY judge quota never loses finished work; re-running resumes. When the judge quota runs
out the run stops cleanly and reports `stopped_early`.
"""
from __future__ import annotations

import json
import os
import statistics

CHECKPOINT = "data/eval_runs/generation_ckpt.jsonl"
CFG_RAW, CFG_FULL = "hybrid+rerank (no guard)", "+guard (full)"


from eval.ragas_metrics import EvalCase, JudgeBudgetExhausted      # noqa: E402  (JudgeBudgetExhausted re-exported)


def is_budget_error(e: BaseException) -> bool:
    """True if `e` (or anything in its cause chain) is a daily-quota failure."""
    from backend.app.llm.groq_llm import LLMRateLimitError
    seen = 0
    while e is not None and seen < 8:
        if isinstance(e, JudgeBudgetExhausted) or (isinstance(e, LLMRateLimitError) and e.daily):
            return True
        if "tokens per day" in str(e) or "(TPD)" in str(e) or "requests per day" in str(e):
            return True
        e = e.__cause__ or e.__context__
        seen += 1
    return False


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def score_case(metrics: dict, test_case, wanted: list[str]) -> dict:
    out: dict = {}
    for name in wanted:
        try:
            metrics[name].measure(test_case)
            out[name] = round(float(metrics[name].score), 3)
        except Exception as e:
            if is_budget_error(e):
                raise JudgeBudgetExhausted(str(e)[:200]) from e
            out[name] = None                      # one failed metric must not kill the run
            out.setdefault("_errors", []).append(f"{name}: {type(e).__name__}")
    return out


def load_checkpoint(path: str) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["case_id"]] = r
                except (json.JSONDecodeError, KeyError):
                    continue                       # torn final line
    return done


def _append(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _summarise(rows: list[dict]) -> dict:
    names = sorted({n for r in rows for n in r if not n.startswith("_")})
    out = {n: _mean([r[n] for r in rows if r.get(n) is not None]) for n in names}
    out["n"] = len(rows)
    return out


def aggregate(records: list[dict]) -> dict:
    """
    `metrics`            : paired comparison on the cases the guarded pipeline ANSWERED (status "ok").
                           An abstention ("Not enough evidence...") is not a faithfulness failure, so it is
                           excluded here and reported as a recall cost in `behaviour` instead.
    `metrics_all_cases`  : every scored answerable case (abstentions included) for transparency.
    """
    all_cfg = {CFG_RAW: [], CFG_FULL: []}
    ans_cfg = {CFG_RAW: [], CFG_FULL: []}
    beh = {"answerable_answered": 0, "answerable_total": 0, "unanswerable_abstained": 0, "unanswerable_total": 0}
    for r in records:
        if r["answerable"]:
            beh["answerable_total"] += 1
            hedged = bool(r.get("hedged"))
            beh["answerable_answered"] += int(r["status"] == "ok" and not hedged)
            beh["answerable_hedged"] = beh.get("answerable_hedged", 0) + int(r["status"] == "ok" and hedged)
            for cfg in all_cfg:
                sc = r.get("scores", {}).get(cfg)
                if sc is not None:
                    all_cfg[cfg].append(sc)
                    if r["status"] == "ok":
                        ans_cfg[cfg].append(sc)
        else:
            beh["unanswerable_total"] += 1
            beh["unanswerable_abstained"] += int(r["status"] in ("abstained", "rejected", "low_confidence"))   # refused OR tagged "not sure"
    ab, ub = beh["answerable_total"], beh["unanswerable_total"]
    beh["answer_rate_on_answerable"] = round(beh["answerable_answered"] / ab, 3) if ab else None
    beh["abstention_rate_on_unanswerable"] = round(beh["unanswerable_abstained"] / ub, 3) if ub else None
    # an answerable question answered only with "the sources don't say" is a false refusal, not a success
    beh["false_refusal_rate_on_answerable"] = round(beh.get("answerable_hedged", 0) / ab, 3) if ab else None
    return {"metrics": {c: _summarise(r) for c, r in ans_cfg.items()},
            "metrics_all_cases": {c: _summarise(r) for c, r in all_cfg.items()},
            "behaviour": beh, "per_case": ans_cfg}


def eval_abstention(pipeline, cases, k: int = 5, checkpoint: str = CHECKPOINT, log=print) -> int:
    """
    Run the unanswerable cases through the pipeline (no judge calls) and checkpoint the outcome.
    Correct behaviour is status "abstained" (or "rejected"). Returns the number of cases newly run.
    """
    done = load_checkpoint(checkpoint)
    n = 0
    for c in (c for c in cases if not c.answerable):
        if c.id in done:
            continue
        full = pipeline.answer(c.question, top_k=k)
        if full.status == "unverified":
            log(f"  {c.id}: LLM unavailable — not recorded (re-run later)")
            continue
        _append(checkpoint, {"case_id": c.id, "question": c.question, "answerable": False, "context": [],
                             "answer": full.answer, "raw_answer": "", "status": full.status, "scores": {}})
        n += 1
        log(f"  {c.id}: {full.status}")
    return n


def rebuild_from_checkpoint(checkpoint: str = CHECKPOINT, golden_total: int | None = None) -> dict:
    """Re-aggregate finished cases without any judge calls. `golden_total` = size of the golden set,
    so reports say "N of M evaluated" instead of implying a complete run."""
    recs = list(load_checkpoint(checkpoint).values())
    out = aggregate(recs)
    n_ans = sum(1 for r in recs if r["answerable"])
    out.update({"stopped_early": None, "cases_scored": len(recs), "cases_planned": golden_total or len(recs),
                "answerable_scored_with_judge": n_ans})
    return out


def eval_generation(pipeline, cases, k: int = 5, limit: int | None = None,
                    wanted_metrics: list[str] | None = None, judge=None,
                    dump_answers: str | None = None, checkpoint: str = CHECKPOINT,
                    log=print) -> dict:
    from eval import runner as R

    if judge is None:
        from eval.ragas_metrics import RagasJudge
        judge = RagasJudge()
    metrics = judge.metrics() if hasattr(judge, "metrics") else R.ragas_metrics(judge)
    answerable = [c for c in cases if c.answerable][:limit]
    unanswerable = [c for c in cases if not c.answerable][:limit]
    done = load_checkpoint(checkpoint)
    if done:
        log(f"resuming: {len(done)} cases already in {checkpoint}")

    stopped = None
    try:
        for c in answerable + unanswerable:
            if c.id in done:
                continue
            if c.answerable:
                chunks = R.retrieve_for_config(pipeline, "hybrid+rerank", c.question, k)
                ctx = [d["text"] for d in chunks]
                raw = R.raw_answer(pipeline, c.question, chunks)
                full = pipeline.answer(c.question, top_k=k)
                scores = {}
                for label, text in ((CFG_RAW, raw), (CFG_FULL, full.answer)):
                    tc = EvalCase(question=c.question, response=text, contexts=ctx, reference=c.reference_answer or "")
                    wanted = list(wanted_metrics or metrics)
                    if not c.reference_answer:                 # correctness / completeness are relative to a reference answer
                        wanted = [m for m in wanted if m not in ("correctness", "completeness")]
                    scores[label] = score_case(metrics, tc, wanted)
                rec = {"case_id": c.id, "question": c.question, "answerable": True, "context": ctx,
                       "answer": full.answer, "raw_answer": raw, "status": full.status, "hedged": bool(getattr(full, "hedged", False)),
                       "scores": scores}
            else:
                full = pipeline.answer(c.question, top_k=k)
                rec = {"case_id": c.id, "question": c.question, "answerable": False, "context": [],
                       "answer": full.answer, "raw_answer": "", "status": full.status, "scores": {}}
            _append(checkpoint, rec)          # only fully scored cases are persisted
            done[c.id] = rec
            log(f"  scored {c.id} ({len(done)} done) status={rec['status']}")
    except JudgeBudgetExhausted as e:
        stopped = f"judge daily quota exhausted: {e}"
        log(f"STOPPED EARLY — {stopped}")

    wanted_ids = {c.id for c in answerable + unanswerable}
    records = [r for cid, r in done.items() if cid in wanted_ids]
    out = aggregate(records)
    out["stopped_early"] = stopped
    out["cases_scored"] = len(records)
    out["cases_planned"] = len(wanted_ids)
    if dump_answers:
        os.makedirs(os.path.dirname(dump_answers) or ".", exist_ok=True)
        with open(dump_answers, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({k_: r[k_] for k_ in ("case_id", "question", "context", "answer",
                                                         "raw_answer", "status")}, ensure_ascii=False) + "\n")
    return out
