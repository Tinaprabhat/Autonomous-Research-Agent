"""
guard_stress.py - a stress test for citation checks that does NOT reuse the tuning data.

    python -m eval.guard_stress build                 # write eval/data/guard_stress_set.jsonl (no models, deterministic)
    python -m eval.guard_stress score [--llm N]       # score every candidate check on every item (cached)
    python -m eval.guard_stress report                # candidate-check comparison: false drops on real supported, catch rate per mutation type
    python -m eval.guard_stress score-final           # run the shipped batched LLM verifier on the same sample
    python -m eval.guard_stress report-final          # the shipped layered guard, per mutation type

Why: the only real negatives are 4 human-labelled unsupported sentences, far too few to tune or compare checks.
This builds extra unsupported sentences by controlled MUTATION of the human-labelled SUPPORTED sentences (so the
label "unsupported" holds by construction, not by an LLM's say-so):
  number_swap    a number in the sentence is replaced by a different one
  antonym_flip   a key word is replaced by its opposite (increases -> decreases, with -> without, ...)
  negation       "It is not the case that ..." prepended (all words still come from the passage)
  qualifier      an unsupported qualifier / extra claim is appended ("... in every setting tested")
  conflation     a clause from a different paper is appended, still citing the original chunk
  wrong_chunk    the sentence is checked against a chunk from a different paper (a wrong citation)
Positives are the real human-labelled supported sentences. Thresholds are chosen on the REAL labelled sentences only
and then applied unchanged to the synthetic items, so the synthetic catch rate is not tuned on.
Caveat: synthetic errors are cleaner than real model errors; treat catch rates as a stress test, not a field rate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.label_validation import DATA, LABEL_SET, _read, load_labels

STRESS_SET = os.path.join(DATA, "guard_stress_set.jsonl")
STRESS_SCORES = os.path.join(DATA, "guard_stress_scores.jsonl")
CATEGORIES = ["number_swap", "antonym_flip", "negation", "qualifier", "conflation", "wrong_chunk"]

_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")
_ANTONYMS = [("increases", "decreases"), ("increase", "decrease"), ("improves", "degrades"), ("improve", "degrade"),
             ("outperforms", "underperforms"), ("outperformed", "underperformed"), ("higher", "lower"),
             ("more", "less"), ("faster", "slower"), ("reduces", "increases"), ("reduced", "increased"),
             ("without", "with"), ("avoids", "requires"), ("larger", "smaller"),
             ("most", "least"), ("stronger", "weaker"), ("fewer", "more"), ("rises", "falls"),
             ("reversed", "reinforced"), ("supports", "contradicts"), ("effective", "ineffective")]
_QUALIFIERS = [", and this holds in every setting that was tested",
               ", outperforming all other baselines by a large margin",
               ", as the authors confirmed in a larger follow-up study from 2019",
               ", while also halving the required training time",
               ", an effect that disappears when the data are noisy"]


def _swap_number(s: str) -> str | None:
    m = _NUM.search(s)
    if not m:
        return None
    tok = m.group(0)
    try:
        val = float(tok.replace(",", "").rstrip("%"))
    except ValueError:
        return None
    new = val * 2 + 7 if val != 0 else 9
    new_s = (f"{new:,.0f}" if "," in tok else (f"{new:.3f}".rstrip("0").rstrip(".") if "." in tok else f"{new:.0f}")) + ("%" if tok.endswith("%") else "")
    return s[:m.start()] + new_s + s[m.end():] if new_s != tok else None


def _flip_antonym(s: str) -> str | None:
    for a, b in _ANTONYMS:
        m = re.search(rf"\b{a}\b", s)
        if m:
            return s[:m.start()] + b + s[m.end():]
    return None


def _lower_first(text: str) -> str:
    """Lower-case a leading plain capitalised word; keep names, acronyms and mixed-case words (LoRA, CARROT) intact."""
    first = text.split(" ", 1)[0]
    return text[0].lower() + text[1:] if first[1:].islower() and first[0].isupper() else text


def _negate(s: str) -> str:
    return "It is not the case that " + _lower_first(s.rstrip(". ")) + "."


def _qualify(s: str, k: int) -> str:
    return s.rstrip(". ") + _QUALIFIERS[k % len(_QUALIFIERS)] + "."


def _conflate(s: str, other: str) -> str:
    return s.rstrip(". ") + ". Furthermore, " + _lower_first(other.rstrip(". ")) + "."


def build(labels_path: str | None = None) -> list[dict]:
    labels_path = labels_path or os.path.join(DATA, "labels_human.jsonl")
    sent_lab, _, _ = load_labels(labels_path)
    cases = _read(LABEL_SET)
    base = []                                   # (case_id, item_id, sentence, supporting chunk text, chunk idx)
    for c in cases:
        for it in c["items"]:
            if sent_lab.get(it["id"]) is True and len(it["sentence"]) > 20:
                from backend.app.citations.guard import lexical_overlap
                best = max(it["refs"], key=lambda n: lexical_overlap(it["sentence"], c["context"][n - 1]))
                base.append({"case": c["case_id"], "id": it["id"], "sentence": it["sentence"],
                             "chunk": c["context"][best - 1]})
    rows = []
    for it in base:
        rows.append({"item": it["id"], "category": "real_supported", "label": "supported",
                     "sentence": it["sentence"], "chunk": it["chunk"]})
    for c in cases:                              # real unsupported (human labelled)
        for it in c["items"]:
            if sent_lab.get(it["id"]) is False:
                from backend.app.citations.guard import lexical_overlap
                best = max(it["refs"], key=lambda n: lexical_overlap(it["sentence"], c["context"][n - 1]))
                rows.append({"item": it["id"], "category": "real_unsupported", "label": "unsupported",
                             "sentence": it["sentence"], "chunk": c["context"][best - 1]})
    n = len(base)
    for i, it in enumerate(base):
        other = base[(i + 7) % n]               # a sentence/chunk from a different case (deterministic rotation)
        if other["case"] == it["case"]:
            other = base[(i + 11) % n]
        muts = {
            "number_swap": _swap_number(it["sentence"]),
            "antonym_flip": _flip_antonym(it["sentence"]),
            "negation": _negate(it["sentence"]),
            "qualifier": _qualify(it["sentence"], i),
            "conflation": _conflate(it["sentence"], other["sentence"]),
        }
        for cat, text in muts.items():
            if text and text != it["sentence"]:
                rows.append({"item": f"{it['id']}~{cat}", "category": cat, "label": "unsupported",
                             "sentence": text, "chunk": it["chunk"]})
        rows.append({"item": f"{it['id']}~wrong_chunk", "category": "wrong_chunk", "label": "unsupported",
                     "sentence": it["sentence"], "chunk": other["chunk"]})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    sc = sub.add_parser("score"); sc.add_argument("--llm", type=int, default=0, help="LLM-verify at most N items per category")
    sub.add_parser("score-final")      # the shipped batched LLM verifier on the same sample
    sub.add_parser("report")
    sub.add_parser("report-final")     # the shipped layered guard
    a = ap.parse_args()
    if a.cmd == "build":
        rows = build()
        with open(STRESS_SET, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        from collections import Counter
        print(len(rows), "items", dict(Counter(r["category"] for r in rows)))
    elif a.cmd == "score":
        from eval.guard_stress_scoring import score_all
        score_all(llm_per_category=a.llm)
    elif a.cmd == "score-final":
        from eval.guard_stress_scoring import score_final
        score_final()
    elif a.cmd == "report-final":
        from eval.guard_stress_scoring import report_final
        report_final()
    else:
        from eval.guard_stress_scoring import report
        report()


if __name__ == "__main__":
    main()
