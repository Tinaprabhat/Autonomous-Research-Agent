"""
answer_quality.py - answer-level faithfulness, label calibration, relevance and completeness for the 30 labelled cases.

    python -m eval.answer_quality [--judge openai/gpt-oss-20b] [--no-judge]

Inputs: eval/data/post_guard_cases.jsonl (written by eval.post_guard_report: for every case the text the USER sees after the
shipped guard, the guard's label and score, and the human sentence labels) and the golden reference answers.

1. FAITHFULNESS (answer level, human labels): raw-answer rate, post-guard rate, and the DeepEval judge's own scores for contrast.
2. LABEL CALIBRATION: does the label the system attaches (ok / low_confidence / hedge) and the guard's score predict whether the
   response is actually free of unsupported claims? Per-label reliability with Wilson intervals, AUROC with a bootstrap interval.
3. RELEVANCE and COMPLETENESS (LLM judge, NOT human-labelled):
     completeness = share of the reference answer's key facts the response states (judge lists the facts and marks each)
     relevance    = share of the response's sentences that directly help answer the question
   The judge is run three times: on the real post-guard responses, on the REFERENCE answers (positive control: should score
   ~1), and on responses shuffled to the wrong question (negative control: should score ~0). If the controls do not separate,
   the scores are reported as not meaningful. Mean scores carry a bootstrap 95% interval over cases.
Reference answers are machine-generated (eval/data/golden_set.json, not human-reviewed), so completeness is "relative to a
machine-written reference".
"""
from __future__ import annotations

import argparse
import json
# [DEAD CODE] unused import (checked 2026-10-15)
# import math
import os
import random
import re
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval import golden
from eval.label_validation import DATA, _read, wilson

CASES = os.path.join(DATA, "post_guard_cases.jsonl")
JUDGMENTS = os.path.join(DATA, "answer_quality_judgments.jsonl")
OUT = os.path.join(DATA, "answer_quality_report.json")
SEED = 17

PROMPT = (
    "You grade a RESPONSE to a QUESTION.\n\nQUESTION:\n{question}\n\n"
    "REFERENCE ANSWER (the key facts a complete answer must contain):\n{reference}\n\nRESPONSE:\n{response}\n\n"
    "Tasks:\n"
    "1. List the distinct key facts in the REFERENCE ANSWER (between 1 and 5 short items). For each, say whether the RESPONSE "
    "states it (covered=true) or clearly implies it. A fact the response does not mention, or contradicts, is covered=false.\n"
    "2. Split the RESPONSE into its sentences and, for each, say whether it is relevant: true only if it directly helps answer "
    "the QUESTION. A sentence that only says the sources lack the answer, or that is off-topic padding, is relevant=false.\n"
    'Reply with exactly one JSON object and nothing else: {{"key_facts": [{{"fact": "...", "covered": true}}], '
    '"sentences": [{{"id": 1, "relevant": true}}]}}')


# ── pure helpers (unit-tested) ──────────────────────────────────────────────────────────────────────────────────

def parse_judgment(text: str) -> dict | None:
    """{"completeness": float, "relevance": float|None, "n_facts": int, "n_sentences": int} or None if unusable."""
    try:
        m = re.search(r"\{.*\}", text or "", re.S)
        data = json.loads(m.group(0))
        facts = [bool(x["covered"]) if isinstance(x["covered"], bool) else str(x["covered"]).lower() == "true"
                 for x in data["key_facts"]]
        sents = [bool(x["relevant"]) if isinstance(x["relevant"], bool) else str(x["relevant"]).lower() == "true"
                 for x in data.get("sentences", [])]
    except (ValueError, KeyError, TypeError, AttributeError):
        return None
    if not facts:
        return None
    return {"completeness": sum(facts) / len(facts), "n_facts": len(facts),
            "relevance": (sum(sents) / len(sents)) if sents else None, "n_sentences": len(sents)}


def bootstrap_mean(values: list[float], n_boot: int = 2000, seed: int = SEED) -> dict:
    if not values:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_boot))
    return {"mean": round(sum(values) / len(values), 3), "ci95": [round(means[int(0.025 * n_boot)], 3), round(means[int(0.975 * n_boot) - 1], 3)],
            "n": len(values)}


def auroc(scores_pos: list[float], scores_neg: list[float]) -> float | None:
    """P(a random 'positive' outranks a random 'negative'); ties count half."""
    if not scores_pos or not scores_neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in scores_pos for q in scores_neg)
    return wins / (len(scores_pos) * len(scores_neg))


def bootstrap_auroc(pos: list[float], neg: list[float], n_boot: int = 2000, seed: int = SEED) -> dict:
    a = auroc(pos, neg)
    if a is None:
        return {"auroc": None, "ci95": [None, None]}
    rng = random.Random(seed)
    vals = sorted(auroc(rng.choices(pos, k=len(pos)), rng.choices(neg, k=len(neg))) for _ in range(n_boot))
    return {"auroc": round(a, 3), "ci95": [round(vals[int(0.025 * n_boot)], 3), round(vals[int(0.975 * n_boot) - 1], 3)]}


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": [lo, hi]}


def calibration(rows: list[dict]) -> dict:
    """Does the label / score predict a response free of unsupported claims? 'correct' = nothing a human marked unsupported is shown."""
    correct = lambda r: not r["shown_unsupported"]
    by_label = {}
    for lab in ("answered", "low_confidence", "false_refusal"):
        g = [r for r in rows if r["status"] == lab]
        by_label[lab] = rate(sum(1 for r in g if correct(r)), len(g))
    good = [r["guard_score"] for r in rows if correct(r)]
    bad = [r["guard_score"] for r in rows if not correct(r)]
    return {"fraction_of_responses_with_no_unsupported_claim_by_label": by_label,
            "n_responses_with_an_unsupported_claim": len(bad),
            "guard_score_predicts_correctness": bootstrap_auroc(good, bad),
            "guard_score_mean_correct": round(sum(good) / len(good), 3) if good else None,
            "guard_score_mean_with_error": round(sum(bad) / len(bad), 3) if bad else None,
            "flagged_low_confidence_among_responses_with_error": rate(
                sum(1 for r in rows if not correct(r) and r["status"] == "low_confidence"), len(bad))}


# ── the LLM judge ───────────────────────────────────────────────────────────────────────────────────────────────────

def judge_once(llm, question: str, reference: str, response: str) -> dict | None:
    from backend.app.llm.groq_llm import LLMError
    prompt = PROMPT.format(question=question, reference=reference, response=response or "(empty response)")
    for attempt in range(4):
        try:
            return parse_judgment(llm.generate(prompt, max_tokens=1800))
        except LLMError as e:
            wait = getattr(e, "retry_after", None)
            if getattr(e, "daily", False):
                raise
            time.sleep(min((wait or 20) + 1, 60))
    return None


def run_judge(model: str, rows: list[dict], refs: dict[str, str]) -> dict[str, dict]:
    from backend.app.llm.groq_llm import GroqLLM
    llm = GroqLLM(model=model, temperature=0.0)
    done = {(r["kind"], r["case_id"]): r for r in _read(JUDGMENTS)}
    out = open(JUDGMENTS, "a", encoding="utf-8")
    ids = [r["case_id"] for r in rows]
    shown = {r["case_id"]: r["shown_text"] for r in rows}
    question = {r["case_id"]: r["question"] for r in rows}
    rot = {cid: ids[(i + 7) % len(ids)] for i, cid in enumerate(ids)}       # a different question's response
    plans = {"real": lambda c: shown[c], "reference": lambda c: refs[c], "shuffled": lambda c: shown[rot[c]]}
    try:
        for kind, text_of in plans.items():
            for cid in ids:
                if (kind, cid) in done:
                    continue
                j = judge_once(llm, question[cid], refs[cid], text_of(cid))
                rec = {"kind": kind, "case_id": cid, "judge": model, **(j or {"unusable": True})}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(f"  {kind}/{cid}: {'unusable' if j is None else round(j['completeness'], 2)}", flush=True)
                time.sleep(8)                                           # stay under the per-minute token cap
    finally:
        out.close()
    return {(r["kind"], r["case_id"]): r for r in _read(JUDGMENTS)}


def summarise_judge(judgments: dict, rows: list[dict]) -> dict:
    res = {}
    for kind in ("real", "reference", "shuffled"):
        js = [judgments[(kind, r["case_id"])] for r in rows if (kind, r["case_id"]) in judgments]
        ok = [j for j in js if not j.get("unusable")]
        res[kind] = {"n_unusable": len(js) - len(ok),
                     "completeness": bootstrap_mean([j["completeness"] for j in ok]),
                     "relevance": bootstrap_mean([j["relevance"] for j in ok if j.get("relevance") is not None])}
    pos, neg = res["reference"], res["shuffled"]
    res["judge_discriminates"] = bool(pos["completeness"]["mean"] is not None and neg["completeness"]["mean"] is not None
                                      and pos["completeness"]["mean"] >= 0.85 and neg["completeness"]["mean"] <= 0.15
                                      and pos["relevance"]["mean"] is not None and neg["relevance"]["mean"] is not None
                                      and pos["relevance"]["mean"] >= 0.85 and neg["relevance"]["mean"] <= 0.25)
    return res


def faithfulness(rows: list[dict]) -> dict:
    n = len(rows)
    raw_ok = sum(1 for r in rows if r["human_answer_faithful_raw"])
    post_ok = sum(1 for r in rows if not r["shown_unsupported"])
    judge = {}
    scores = os.path.join(DATA, "label_scores.jsonl")
    if os.path.exists(scores):
        js = {r["case_id"]: r["judge_faithfulness"] for r in _read(scores) if r.get("judge_faithfulness") is not None}
        vals = [js[r["case_id"]] for r in rows if r["case_id"] in js]
        judge = {"deepeval_faithfulness_mean": round(sum(vals) / len(vals), 3) if vals else None,
                 "deepeval_faithfulness_min": min(vals) if vals else None, "n": len(vals)}
    return {"raw_answer_faithful_by_human": rate(raw_ok, n),
            "post_guard_response_free_of_unsupported_claims_by_human": rate(post_ok, n), "deepeval_judge_for_contrast": judge}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default="openai/gpt-oss-20b")
    ap.add_argument("--no-judge", action="store_true", help="only the faithfulness and calibration parts (no LLM calls)")
    a = ap.parse_args()
    rows = _read(CASES)
    if len(rows) != 30 or any("shown_text" not in r for r in rows):
        sys.exit("run `python -m eval.post_guard_report` first (needs shown_text for all 30 cases)")
    refs = {c.id: c.reference_answer for c in golden.load(only_reviewed=False)}
    res = {"n_cases": len(rows), "faithfulness": faithfulness(rows), "label_calibration": calibration(rows)}
    if not a.no_judge:
        judgments = run_judge(a.judge, rows, refs)
        res["relevance_completeness"] = {"judge_model": a.judge, **summarise_judge(judgments, rows)}
        cases = []
        for r in rows:
            j = judgments.get(("real", r["case_id"]), {})
            cases.append({"case_id": r["case_id"], "label": r["status"], "guard_score": r["guard_score"],
                          "human_faithful_raw": r["human_answer_faithful_raw"], "unsupported_shown": len(r["shown_unsupported"]),
                          "completeness": j.get("completeness"), "relevance": j.get("relevance")})
        with open(os.path.join(DATA, "answer_quality_cases.jsonl"), "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c) + "\n")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
