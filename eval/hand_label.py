"""
hand_label.py — terminal tool for hand-labelling answer faithfulness, one keystroke per case.

    python -m eval.hand_label data/eval_runs/answers_<ts>.jsonl [--out data/eval_runs/labels.jsonl]

For each case it prints the question, the retrieved context and the generated answer, then asks
    faithful? (y/n):
and reads a single key (no Enter). Every verdict is appended to the results file and fsync'd
immediately — a crash or Ctrl-C loses at most the case currently on screen. On launch, cases
whose case_id is already in the results file are skipped, so 200 cases can be labelled over
several sittings. Press q to quit.

Results file: one JSON object per line: {"case_id", "verdict", "timestamp"}  (verdict: "y" | "n").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

DEFAULT_OUT = "data/eval_runs/hand_labels.jsonl"
VALID = {"y", "n"}


def read_key() -> str:
    """One keypress, no Enter. Windows: msvcrt; POSIX: raw tty."""
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):      # special key prefix
            msvcrt.getwch()
            return ""
        return ch
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def load_labelled(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["case_id"])
            except (json.JSONDecodeError, KeyError):
                continue            # tolerate a torn last line from a crash
    return done


def append_label(path: str, case_id: str, verdict: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rec = {"case_id": case_id, "verdict": verdict,
           "timestamp": datetime.now(timezone.utc).isoformat()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def show(case: dict, i: int, total: int, out=print) -> None:
    out("\n" + "=" * 78)
    out(f"[{i}/{total}]  {case['case_id']}")
    out("QUESTION: " + case["question"])
    out("\nRETRIEVED CONTEXT:")
    for n, c in enumerate(case.get("context", []), start=1):
        out(f"  [{n}] {c[:600]}")
    out("\nANSWER:\n  " + (case.get("answer") or "(empty)"))
    out("")


def run(cases_path: str, out_path: str = DEFAULT_OUT, getkey=read_key, out=print) -> int:
    """Returns the number of cases labelled in this session."""
    cases = load_cases(cases_path)
    done = load_labelled(out_path)
    todo = [(i, c) for i, c in enumerate(cases, start=1) if c["case_id"] not in done]
    out(f"{len(done)} already labelled, {len(todo)} remaining of {len(cases)}.")
    n = 0
    try:
        for i, c in todo:
            show(c, i, len(cases), out)
            while True:
                out("faithful? (y/n, q=quit): ")
                key = (getkey() or "").lower()
                if key in VALID:
                    append_label(out_path, c["case_id"], key)
                    out(key)
                    n += 1
                    break
                if key in ("q", "\x03"):                # q or Ctrl-C delivered as a key
                    out("quit")
                    return n
    except KeyboardInterrupt:
        out("\ninterrupted — progress saved.")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", help="answers .jsonl produced by `python -m eval.runner ... --dump-answers`")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    n = run(a.cases, a.out)
    done = load_labelled(a.out)
    print(f"\nLabelled {n} this session; {len(done)} total in {a.out}")


if __name__ == "__main__":
    main()
