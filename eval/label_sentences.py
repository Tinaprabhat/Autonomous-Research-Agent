"""
label_sentences.py - terminal tool for the human labelling that validates the judge and the NLI threshold.

    python -m eval.label_sentences --labeler yourname [--out eval/data/labels_human.jsonl]

For each of the 30 answers it shows the question, the retrieved passages and the model answer, then asks
  * for every cited sentence:  "supported by the passage(s) it cites?  (y/n)"
  * once per answer:           "whole answer faithful to the passages?  (y/n)"
One keystroke each (no Enter). Every verdict is fsync'd immediately; re-running skips what is already labelled,
so it can be done over several sittings. q quits. The label set contains no model scores (blind labelling).

Rubric
  supported (sentence): at least one cited passage states or clearly implies the sentence's claim, including every
                        number/name in it. Paraphrase is fine; extra detail the passage does not give is NOT.
  faithful (answer):    every factual claim in the answer (cited or not) is supported by the retrieved passages.
                        An answer that says "the sources do not contain this" is faithful if that is true.
Then:  python -m eval.label_validation report --labels eval/data/labels_human.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.hand_label import read_key
from eval.label_validation import DATA, LABEL_SET, _read

DEFAULT_OUT = os.path.join(DATA, "labels_human.jsonl")


def _append(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def ask(prompt, getkey, out):
    while True:
        out(prompt + " (y/n, q=quit): ")
        k = (getkey() or "").lower()
        if k in ("y", "n"):
            out(k)
            return k == "y"
        if k in ("q", "\x03"):
            return None


def run(labeler: str, out_path: str = DEFAULT_OUT, getkey=read_key, out=print) -> int:
    cases = _read(LABEL_SET)
    have = {(r["type"], r["id"]) for r in _read(out_path)}
    n = 0
    for ci, c in enumerate(cases, 1):
        todo_items = [it for it in c["items"] if ("sentence", it["id"]) not in have]
        need_answer = ("answer", c["case_id"]) not in have
        if not todo_items and not need_answer:
            continue
        out("\n" + "=" * 78)
        out(f"[{ci}/{len(cases)}] {c['case_id']}   QUESTION: {c['question']}")
        for i, p in enumerate(c["context"], 1):
            out(f"  [{i}] {p}")
        out("\nANSWER:\n  " + c["answer"] + "\n")
        for it in todo_items:
            out(f"SENTENCE (cites {it['refs']}): {it['sentence']}")
            v = ask("supported by the passage(s) it cites?", getkey, out)
            if v is None:
                return n
            _append(out_path, {"type": "sentence", "id": it["id"], "supported": v, "labeler": labeler,
                               "ts": datetime.now(timezone.utc).isoformat()})
            n += 1
        if need_answer:
            v = ask("WHOLE answer faithful to the passages?", getkey, out)
            if v is None:
                return n
            _append(out_path, {"type": "answer", "id": c["case_id"], "faithful": v, "labeler": labeler,
                               "ts": datetime.now(timezone.utc).isoformat()})
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeler", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    n = run(a.labeler, a.out)
    print(f"\nlabelled {n} items this session -> {a.out}")


if __name__ == "__main__":
    main()
