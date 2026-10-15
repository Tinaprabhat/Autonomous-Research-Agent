"""
run_all.py — unattended end-to-end evaluation driver. Safe to run in the background:

    python -m eval.run_all --local > data/eval_runs/run_all.log 2>&1

Stages (each logged to data/eval_runs/run_all_status.json so progress survives interruptions and a
re-run resumes where it stopped; a failed stage is recorded, not fatal, unless later stages need it):
  1. golden   build the 100-case verified golden set        (skipped if eval/data/golden_set.json exists)
  2. retrieval  recall@k / MRR ablation                     (no LLM cost)
  3. generation RAGAS metrics (faithfulness, hallucination, correctness, completeness): with vs without the guard (Groq judge)
  4. dashboard  bake eval/reports/dashboard.html
Each stage runs in its own subprocess so the embedded Qdrant lock is released between stages.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

STATUS = "data/eval_runs/run_all_status.json"
# Models are cached locally; skipping huggingface_hub's per-load online check removes a network failure
# mode (a single failed request permanently breaks its HTTP client for the process).
OFFLINE_ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def load_status() -> dict:
    try:
        with open(STATUS, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_status(st: dict) -> None:
    os.makedirs(os.path.dirname(STATUS), exist_ok=True)
    with open(STATUS, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def run_stage(name: str, cmd: list[str], st: dict, retries: int = 2) -> bool:
    if st.get(name, {}).get("ok"):
        print(f"[{name}] already done, skipping", flush=True)
        return True
    for attempt in range(1, retries + 2):
        print(f"\n===== [{name}] attempt {attempt}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd, env=OFFLINE_ENV)
        paused = rc == 3        # a model's DAILY free-tier quota is exhausted: retrying today is pointless
        st[name] = {"ok": rc == 0, "rc": rc, "seconds": round(time.time() - t0), "attempts": attempt,
                    "paused_daily_quota": paused}
        save_status(st)
        if rc == 0:
            return True
        if paused:
            print(f"[{name}] PAUSED: daily quota exhausted. Re-run later; it resumes from its checkpoint.", flush=True)
            return False
        time.sleep(20)          # let rate limits / locks clear before retrying
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--gen-limit", type=int, default=30, help="answerable cases for LLM-judged metrics")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--allow-partial-golden", action="store_true",
                    help="continue with whatever golden cases verified if the quota runs out")
    a = ap.parse_args()
    partial = ["--allow-partial"] if a.allow_partial_golden else []
    py = sys.executable
    local = ["--local"] if a.local else []
    st = load_status()

    if not os.path.exists("eval/data/golden_set.json"):
        run_stage("golden", [py, "-m", "eval.build_golden_set", *local, *partial], st)
    else:
        st["golden"] = {"ok": True, "note": "existing golden_set.json"}
    if not os.path.exists("eval/data/golden_set.json"):
        print("FATAL: no golden set — cannot evaluate.", flush=True)
        sys.exit(1)

    run_stage("retrieval", [py, "-m", "eval.runner", "retrieval", "--k", str(a.k), *local], st)
    run_stage("generation", [py, "-m", "eval.runner", "generation", "--k", str(a.k),
                             "--limit", str(a.gen_limit), "--dump-answers", *local], st, retries=1)
    run_stage("dashboard", [py, "-m", "eval.dashboard"], st, retries=0)
    save_status(st)
    ok = all(v.get("ok") for v in st.values())
    print("\nSTATUS:", json.dumps(st), flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
