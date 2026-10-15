"""
ragas_eval.py - RAGAS faithfulness / hallucination / correctness / completeness on the 30 human-labelled cases.

    python -m eval.ragas_eval run [--judge openai/gpt-oss-120b] [--stage validate|guarded|all]
    python -m eval.ragas_eval report

Stage "validate": RAGAS faithfulness on the RAW (unguarded) answers, checked against the human answer labels. This is the test
that DeepEval failed (it scored every answer 1.0): does the metric score the human-labelled unfaithful answers lower?
Stage "guarded": faithfulness, correctness (F1 of FactualCorrectness precision and reference-claim coverage) and completeness
(coverage: share of the reference's claims the response states) on the text the user sees after the shipped guard (eval/data/post_guard_cases.jsonl); hallucination = 1 - faithfulness. When the guard changed
nothing, the guarded text equals the raw answer and the validated faithfulness score is reused (no second judge call).

Every judge score is appended to eval/data/ragas_eval_scores.jsonl, so an interrupted or quota-limited run resumes where it
stopped (exit code 3 = the judge's daily token quota ran out; run again after it resets). All means carry bootstrap 95%
intervals over cases; with n = 30 they are wide. Reference answers are machine-written (not human-reviewed).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval import golden
from eval.answer_quality import bootstrap_auroc, bootstrap_mean, rate
from eval.label_validation import DATA, LABEL_SET, _read
from eval.ragas_metrics import DEFAULT_JUDGE, EvalCase, JudgeBudgetExhausted, RagasJudge, f1

SCORES = os.path.join(DATA, "ragas_eval_scores.jsonl")
CASES = os.path.join(DATA, "post_guard_cases.jsonl")
REPORT = os.path.join(DATA, "ragas_eval_report.json")


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", re.sub(r"\[\d+\]", "", text or "").lower()).strip()


def load_scores() -> dict[tuple[str, str, str], dict]:
    return {(r["kind"], r["case_id"], r["metric"]): r for r in _read(SCORES)}


def get(scores: dict, kind: str, cid: str, metric: str):
    r = scores.get((kind, cid, metric))
    return None if r is None else r["value"]


def inputs() -> list[dict]:
    """One dict per labelled case: question, contexts, raw answer, guarded text, reference, human labels, guard label."""
    refs = {c.id: c.reference_answer for c in golden.load(only_reviewed=False)}
    label_set = {c["case_id"]: c for c in _read(LABEL_SET)}
    human = {r["id"]: bool(r["faithful"]) for r in _read(os.path.join(DATA, "labels_human.jsonl")) if r["type"] == "answer"}
    out = []
    for r in _read(CASES):
        c = label_set[r["case_id"]]
        out.append({"case_id": r["case_id"], "question": c["question"], "contexts": c["context"], "raw": c["answer"],
                    "guarded": r["shown_text"], "reference": refs.get(r["case_id"], ""), "human_faithful": human[r["case_id"]],
                    "status": r["status"], "unsupported_shown": len(r["shown_unsupported"]),
                    "same_text": norm(r["shown_text"]) == norm(c["answer"])})
    return out


def run(judge_model: str, stage: str) -> int:
    rows = inputs()
    # validation order: the 5 human-unfaithful answers and 5 faithful ones first, so the key result arrives early
    first = [r for r in rows if not r["human_faithful"]] + [r for r in rows if r["human_faithful"] and r["status"] == "answered"][:5]
    order = first + [r for r in rows if r not in first]
    judge = RagasJudge(judge_model)
    scores = load_scores()
    out = open(SCORES, "a", encoding="utf-8")

    def record(kind, cid, metric, value):
        rec = {"kind": kind, "case_id": cid, "metric": metric, "value": value, "judge": judge_model, "ts": int(time.time())}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        scores[(kind, cid, metric)] = rec

    def score(kind, r, text, needed):
        tc = EvalCase(r["question"], text, r["contexts"], r["reference"])
        pending = {m for m in needed if (kind, r["case_id"], m) not in scores}
        if not pending:
            return
        got = judge.raw(tc, pending)
        for m in pending:
            record(kind, r["case_id"], m, got[m])

    try:
        if stage in ("validate", "all"):
            for r in order:
                score("raw", r, r["raw"], {"faithfulness"})
                print(f"  raw {r['case_id']}: faithfulness {get(scores, 'raw', r['case_id'], 'faithfulness'):.3f} "
                      f"(human faithful={r['human_faithful']}) tokens so far {judge.tokens}", flush=True)
        if stage in ("guarded", "all"):
            for r in order:
                if r["same_text"] and get(scores, "raw", r["case_id"], "faithfulness") is not None \
                        and ("guarded", r["case_id"], "faithfulness") not in scores:
                    record("guarded", r["case_id"], "faithfulness", get(scores, "raw", r["case_id"], "faithfulness"))
                score("guarded", r, r["guarded"], {"faithfulness", "fact_precision", "fact_coverage"})
                print(f"  guarded {r['case_id']}: faith {get(scores, 'guarded', r['case_id'], 'faithfulness'):.3f} "
                      f"prec {get(scores, 'guarded', r['case_id'], 'fact_precision'):.3f} "
                      f"coverage {get(scores, 'guarded', r['case_id'], 'fact_coverage'):.3f} tokens so far {judge.tokens}", flush=True)
    except JudgeBudgetExhausted as e:
        print(f"STOPPED: judge daily token quota exhausted ({str(e)[:120]}). Re-run later to resume.", flush=True)
        return 3
    finally:
        out.close()
        judge.close()
    return 0


def report() -> dict:
    rows, scores = inputs(), load_scores()
    res: dict = {"n_cases": len(rows)}

    # -- validation: does faithfulness separate the human labels? --------------------------------------------------
    raw = [(r, get(scores, "raw", r["case_id"], "faithfulness")) for r in rows]
    raw = [(r, v) for r, v in raw if v is not None]
    unf = [v for r, v in raw if not r["human_faithful"]]
    fai = [v for r, v in raw if r["human_faithful"]]
    val = {"n_scored_raw": len(raw), "n_human_unfaithful": len(unf), "n_human_faithful": len(fai),
           "faithfulness_unfaithful_answers": {r["case_id"]: round(v, 3) for r, v in raw if not r["human_faithful"]},
           "mean_faithfulness_unfaithful": round(sum(unf) / len(unf), 3) if unf else None,
           "mean_faithfulness_faithful": round(sum(fai) / len(fai), 3) if fai else None}
    if unf and fai:
        val["unfaithful_flagged_by_score_below_1"] = rate(sum(1 for v in unf if v < 1.0), len(unf))
        val["faithful_wrongly_flagged_by_score_below_1"] = rate(sum(1 for v in fai if v < 1.0), len(fai))
        val["auroc_faithful_answers_outscore_unfaithful"] = bootstrap_auroc(fai, unf)
    diag = os.path.join(DATA, "deepeval_diagnosis.json")
    if os.path.exists(diag):
        d = json.load(open(diag, encoding="utf-8"))
        bad = [x for x in d if not x["human_faithful"]]
        val["deepeval_same_cases_for_contrast"] = {
            "cases": len(d), "unfaithful_flagged_default": sum(1 for x in bad if x["default_score"] < 1),
            "unfaithful_flagged_penalize_ambiguous": sum(1 for x in bad if x["strict_score_yes_only"] < 1), "n_unfaithful": len(bad)}
    res["validation_raw_answers_vs_human_labels"] = val

    # -- the four metrics on what the user sees after the guard ------------------------------------------------------------
    # faithfulness / hallucination: judged directly, or (when the guard changed nothing, so the text and contexts are identical to the
    # validated raw answer) taken from the raw score; correctness / completeness: judged cases only
    g = {"faithfulness": [], "hallucination": [], "correctness": [], "completeness": []}
    per_case = {}
    for r in rows:
        cid = r["case_id"]
        f = get(scores, "guarded", cid, "faithfulness")
        source = "judged"
        if f is None and r["same_text"]:
            f, source = get(scores, "raw", cid, "faithfulness"), "reused_raw_score"
        p, rc = get(scores, "guarded", cid, "fact_precision"), get(scores, "guarded", cid, "fact_coverage")
        legacy = get(scores, "guarded", cid, "fact_recall")
        d = {"status": r["status"], "faithfulness_source": source if f is not None else None}
        if f is not None:
            d["faithfulness"], d["hallucination"] = f, 1.0 - f
        if p is not None and rc is not None:
            d["correctness"], d["completeness"] = f1(p, rc), rc
        if legacy is not None:
            d["ragas_recall_legacy"] = legacy
        per_case[cid] = d
        for k in g:
            if k in d:
                g[k].append(d[k])
    res["guarded_responses"] = {k: bootstrap_mean(v) for k, v in g.items()}
    legacy_vals = [d["ragas_recall_legacy"] for d in per_case.values() if "ragas_recall_legacy" in d]
    res["ragas_recall_legacy_for_contrast"] = {**bootstrap_mean(legacy_vals), "zeros": sum(1 for v in legacy_vals if v == 0)}
    prec_vals = [get(scores, "guarded", r["case_id"], "fact_precision") for r in rows if get(scores, "guarded", r["case_id"], "fact_precision") is not None]
    res["precision_vs_reference"] = {**bootstrap_mean(prec_vals), "zeros": sum(1 for v in prec_vals if v == 0)}
    halluc = [v for v in g["hallucination"]]
    res["guarded_responses"]["responses_with_any_unsupported_statement"] = rate(sum(1 for v in halluc if v > 0), len(halluc)) if halluc else None
    res["coverage"] = {"cases": len(rows), "with_faithfulness_and_hallucination": len(g["faithfulness"]),
                       "with_correctness_and_completeness": len(g["correctness"]),
                       "pending_because_judge_daily_quota": [r["case_id"] for r in rows if "correctness" not in per_case[r["case_id"]]]}
    by_label = {}
    for cid, d in per_case.items():
        lab = by_label.setdefault(d["status"], {"n": 0, **{k: [] for k in g}})
        lab["n"] += 1
        for k in g:
            if k in d:
                lab[k].append(d[k])
    res["guarded_by_label"] = {lab: {"n": v["n"], **{k: (round(sum(v[k]) / len(v[k]), 3) if v[k] else None) for k in g},
                                     **{f"n_{k}": len(v[k]) for k in ("faithfulness", "correctness")}} for lab, v in by_label.items()}
    clean = [d["faithfulness"] for cid, d in per_case.items() if "faithfulness" in d and not next(r for r in rows if r["case_id"] == cid)["unsupported_shown"]]
    dirty = [d["faithfulness"] for cid, d in per_case.items() if "faithfulness" in d and next(r for r in rows if r["case_id"] == cid)["unsupported_shown"]]
    if clean and dirty:
        res["guarded_faithfulness_vs_human_unsupported_shown"] = {"n_clean": len(clean), "n_with_unsupported": len(dirty),
                                                                   "auroc": bootstrap_auroc(clean, dirty)}
    res["per_case"] = per_case
    judges = sorted({r["judge"] for r in scores.values()})
    res["judge_models"] = judges
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--judge", default=DEFAULT_JUDGE)
    r.add_argument("--stage", choices=["validate", "guarded", "all"], default="all")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "run":
        sys.exit(run(a.judge, a.stage))
    report()


if __name__ == "__main__":
    main()
