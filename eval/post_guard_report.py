"""
post_guard_report.py - what does the USER see after the shipped citation guard? (error / answer / false-refusal rates)

    python -m eval.post_guard_report [--model openai/gpt-oss-120b] [--out eval/data/post_guard_report.json]

Data: the 30 answerable golden cases whose raw (unguarded) model answers were labelled sentence by sentence by a human
(eval/data/label_set.jsonl, labels_human.jsonl). Each cached raw answer is replayed through the SHIPPED guard
(guard.check_citations with the LLM verifier, layer A on) and classified exactly like ResearchPipeline._assemble:

  answered        status "ok" and not hedge-only: the user sees only the sentences the guard kept
  false refusal   status "ok" but hedge-only ("the sources don't say"): on an answerable question, a non-answer
  low_confidence  more than half of the checkable sentences dropped: the user sees the model's ORIGINAL answer with a
                  "not sure" tag (so unsupported sentences in it ARE visible, but flagged)

Rates (all with Wilson 95% intervals):
  answer rate               answered / n
  false-refusal rate        hedge-only / n
  low-confidence rate       low_confidence / n
  post-guard error rate     answered responses that still contain a human-labelled unsupported sentence / answered
  unflagged-error exposure  the same count / n
  any-visible-error rate    responses (tagged or not) showing an unsupported sentence / n   (compare: pre-guard 5/30)

Limits (also printed): n = 30 so intervals are wide; only 5 answers contained an unsupported sentence; thresholds were set on
the same labelled sentences (in-sample for layer A); the replay cannot reproduce the pipeline's "you must cite" retry for
an answer with no citation (a007), which is therefore counted as low_confidence; the verifier model is the one named in the
output (gpt-oss-120b, the model that was evaluated) and no fallback is used.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.citations.guard import EntailmentChain, _split_sentences, check_citations
from backend.app.citations.llm_verifier import LLMEntailment
from eval.label_validation import DATA, LABEL_SET, _read, wilson

OUT = os.path.join(DATA, "post_guard_cases.jsonl")


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", re.sub(r"\[\d+\]", "", text).lower()).strip()


def human_sentence_verdicts(path: str) -> dict[str, bool]:
    """normalised sentence text -> human 'supported'. Includes the two items that were splitter fragments: their stored
    verdict is the human's verdict on the full sentence, and the fragment text is a substring of it (see lookup())."""
    labels = {}
    for r in _read(path):
        if r["type"] == "sentence":
            labels[r["id"]] = bool(r["supported"])
    out = {}
    for case in _read(LABEL_SET):
        for it in case["items"]:
            if it["id"] in labels:
                out[norm(it["sentence"])] = labels[it["id"]]
    return out


def lookup(verdicts: dict[str, bool], sentence: str) -> bool | None:
    n = norm(sentence)
    if n in verdicts:
        return verdicts[n]
    for frag, v in verdicts.items():                 # a cached splitter fragment of this (now whole) sentence
        if len(frag) > 15 and frag in n:
            return v
    return None


def classify(guard) -> str:
    if guard.abstain:
        return "low_confidence"
    if guard.passthrough_sentences and not guard.kept_sentences:
        return "false_refusal"
    return "answered"


def run(model: str, labels_path: str) -> list[dict]:
    from backend.app.llm.groq_llm import GroqLLM
    done = {r["case_id"]: r for r in _read(OUT)}
    verdicts = human_sentence_verdicts(labels_path)
    answer_ok = {r["id"]: bool(r["faithful"]) for r in _read(labels_path) if r["type"] == "answer"}
    chain = EntailmentChain(llm=LLMEntailment(GroqLLM(model=model)), mode="llm")
    out = open(OUT, "a", encoding="utf-8")
    try:
        for case in _read(LABEL_SET):
            cid = case["case_id"]
            if cid in done:
                continue
            chunks = [{"text": t} for t in case["context"]]
            for attempt in range(4):                      # the free tier caps tokens per MINUTE: pace, then retry
                g = check_citations(case["answer"], chunks, chain)
                if g.verification != "lexical_only":
                    break
                print(f"  {cid}: verifier unavailable (attempt {attempt + 1}); waiting", flush=True)
                time.sleep(25)
            status = classify(g)
            if status == "low_confidence":
                shown = _split_sentences(case["answer"])           # the original answer is shown, tagged
            else:
                shown = g.kept_sentences + g.passthrough_sentences
            shown_labels = [(s, lookup(verdicts, s)) for s in shown]
            shown_text = " ".join(shown).strip()
            rec = {"case_id": cid, "question": case["question"], "shown_text": shown_text,
                   "guard_score": g.faithfulness_score, "status": status, "verification": g.verification,
                   "verifier_failed": g.verification == "lexical_only",
                   "n_sentences": len(_split_sentences(case["answer"])), "kept": len(g.kept_sentences),
                   "dropped": len(g.dropped_sentences), "passthrough": len(g.passthrough_sentences),
                   "human_answer_faithful_raw": answer_ok.get(cid),
                   "shown_unsupported": [s for s, v in shown_labels if v is False],
                   "shown_unlabelled": [s for s, v in shown_labels if v is None and not any(
                       d["decision"] == "passthrough" and d["sentence"] == s for d in g.details)],
                   "dropped_but_human_supported": [d["sentence"] for d in g.details
                                                   if d["decision"] == "dropped" and lookup(verdicts, d["sentence"]) is True],
                   "dropped_and_human_unsupported": [d["sentence"] for d in g.details
                                                     if d["decision"] == "dropped" and lookup(verdicts, d["sentence"]) is False]}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  {cid}: {status} (verification={g.verification or 'not needed'})", flush=True)
            if g.verification:
                time.sleep(12)                            # ~4 verifier calls/min stays under the 8,000 TPM cap
    finally:
        out.close()
    return _read(OUT)


def ci(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": [lo, hi]}


def summarize(rows: list[dict], model: str) -> dict:
    bad = [r["case_id"] for r in rows if r.get("verifier_failed")]
    rows = [r for r in rows if not r.get("verifier_failed")]          # a degraded (lexical_only) replay is not the shipped guard
    n = len(rows)
    answered = [r for r in rows if r["status"] == "answered"]
    unflagged_err = [r for r in answered if r["shown_unsupported"]]
    visible_err = [r for r in rows if r["shown_unsupported"]]
    pre_err = [r for r in rows if r["human_answer_faithful_raw"] is False]
    caught = [r for r in pre_err if not r["shown_unsupported"] or r["status"] == "low_confidence"]
    res = {
        "verifier_model": model, "n_answers": n,
        "excluded_because_the_verifier_never_ran": bad,
        "pre_guard_error_rate": ci(len(pre_err), n),
        "answer_rate": ci(len(answered), n),
        "false_refusal_rate": ci(sum(1 for r in rows if r["status"] == "false_refusal"), n),
        "low_confidence_rate": ci(sum(1 for r in rows if r["status"] == "low_confidence"), n),
        "post_guard_error_rate_among_answered": ci(len(unflagged_err), len(answered)),
        "unflagged_error_exposure_of_all": ci(len(unflagged_err), n),
        "any_visible_error_rate_of_all": ci(len(visible_err), n),
        "erroneous_answers_not_silently_shown": ci(len(caught), len(pre_err)),
        "correct_answers_wrongly_tagged_low_confidence": ci(
            sum(1 for r in rows if r["human_answer_faithful_raw"] and r["status"] == "low_confidence"),
            sum(1 for r in rows if r["human_answer_faithful_raw"])),
        "unsupported_sentences_removed_by_the_guard": ci(
            sum(len(r["dropped_and_human_unsupported"]) for r in rows),
            sum(len(r["dropped_and_human_unsupported"]) + len(r["shown_unsupported"]) for r in rows)),
        "unlabelled_shown_sentences": sum(len(r["shown_unlabelled"]) for r in rows),
        "supported_sentences_wrongly_dropped": sum(len(r["dropped_but_human_supported"]) for r in rows),
        "unsupported_sentences_dropped": sum(len(r["dropped_and_human_unsupported"]) for r in rows),
        "cases": {r["case_id"]: r["status"] for r in rows},
        "answers_with_unflagged_unsupported_sentence": [r["case_id"] for r in unflagged_err],
    }
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--labels", default=os.path.join(DATA, "labels_human.jsonl"))
    ap.add_argument("--out", default=os.path.join(DATA, "post_guard_report.json"))
    a = ap.parse_args()
    rows = run(a.model, a.labels)
    res = summarize(rows, a.model)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
