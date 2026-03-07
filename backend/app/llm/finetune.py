"""
finetune.py
-----------
Local LLM optimizer for CPU-only machines (Lenovo ThinkBook).
Auto-detects CPU cores and tunes Ollama params via binary search.
"""

import os
import sys
import json
import time
import subprocess
import requests
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from backend.app.llm.local_llm import LocalLLM

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL     = "http://localhost:11434"
OUTPUT_DIR     = "data/finetuned_model"
MODELFILE_PATH = os.path.join(OUTPUT_DIR, "Modelfile")
CUSTOM_MODEL   = "research-mistral"
BASELINE_MODEL = "mistral"

BENCHMARK_PROMPT = (
    "Answer using ONLY this context: "
    "'Few-shot learning uses data augmentation, transfer learning, and metric learning. "
    "Metric learning trains embeddings so similar classes are close and dissimilar ones far apart.'\n"
    "Question: List the strategies for few-shot learning and explain metric learning.\nAnswer:"
)

SYSTEM_PROMPT = """\
You are a concise research assistant. Rules:
1. Answer using ONLY the provided context.
2. Keep answers under 3 sentences. For list+explain questions, use inline format.
3. If context lacks the answer say: "The context does not contain this information."
4. No preamble. Start your answer immediately.
5. Never use bullet points or numbered lists. Write lists inline separated by commas.

Example of correct list+explain format:
Q: List the methods and explain dropout.
A: The methods are data augmentation, dropout, and weight decay. Dropout randomly disables neurons during training to prevent overfitting.\
"""

# ── Auto-detect CPU ───────────────────────────────────────────────────────────

def detect_cpu() -> dict:
    physical = multiprocessing.cpu_count()
    # Ollama performs best with physical cores, not hyperthreads
    # Conservative: use 75% of logical cores, min 2
    recommended_threads = max(2, int(physical * 0.75))
    print(f"  CPU logical cores : {physical}")
    print(f"  Recommended threads: {recommended_threads}")
    return {"logical": physical, "recommended": recommended_threads}


# ── Param Grid Search ─────────────────────────────────────────────────────────

def time_config(model: str, num_ctx: int, num_thread: int, num_predict: int) -> float | None:
    """Create a temp Modelfile, register it, time one response, clean up."""
    tmp_name = "research-mistral-tmp"
    tmp_path = os.path.join(OUTPUT_DIR, "Modelfile_tmp")

    content = f"""\
FROM {model}
SYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"
PARAMETER num_predict {num_predict}
PARAMETER num_ctx {num_ctx}
PARAMETER num_thread {num_thread}
PARAMETER temperature 0.3
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1
PARAMETER stop "</s>"
PARAMETER stop "[INST]"
"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(tmp_path, "w") as f:
        f.write(content)

    # Register temp model
    r = subprocess.run(
        ["ollama", "create", tmp_name, "-f", tmp_path],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        return None

    # Time it (single run — we want wall-clock, not average)
    try:
        llm = LocalLLM(model=tmp_name, temperature=0.3, max_tokens=num_predict, timeout=120)
        t0 = time.time()
        llm.generate(BENCHMARK_PROMPT)
        latency = round(time.time() - t0, 2)
    except Exception:
        latency = None

    # Delete temp model
    subprocess.run(["ollama", "rm", tmp_name], capture_output=True)
    return latency


def grid_search(cpu: dict) -> dict:
    """
    Tests a small grid of (num_ctx, num_thread) combos and returns the best.
    num_predict is fixed at 150 — it's always a win to cap output tokens.
    """
    print("\n[2/4] Running param grid search (this takes a few minutes)...")

    t = cpu["recommended"]
    # Test thread counts around the recommended value
    thread_candidates = sorted(set([
        max(2, t - 2),
        t,
        min(cpu["logical"], t + 2),
    ]))

    # ctx: 512 is faster but may truncate; 2048 is safe; 1024 is middle
    ctx_candidates = [512, 1024, 2048]

    best = {"latency": float("inf"), "num_ctx": 2048, "num_thread": t, "num_predict": 120}
    results = []

    for num_ctx in ctx_candidates:
        for num_thread in thread_candidates:
            print(f"  Testing num_ctx={num_ctx}, num_thread={num_thread} ...", end=" ", flush=True)
            latency = time_config(BASELINE_MODEL, num_ctx, num_thread, num_predict=120)
            if latency is not None:
                print(f"{latency}s")
                results.append((latency, num_ctx, num_thread))
                if latency < best["latency"]:
                    best = {"latency": latency, "num_ctx": num_ctx,
                            "num_thread": num_thread, "num_predict": 120}
            else:
                print("failed")

    print(f"\n  Best config: num_ctx={best['num_ctx']}, "
          f"num_thread={best['num_thread']} -> {best['latency']}s")
    return best


# ── Write Final Modelfile ─────────────────────────────────────────────────────

def write_modelfile(best: dict):
    content = f"""\
FROM {BASELINE_MODEL}

SYSTEM \"\"\"\
{SYSTEM_PROMPT}
\"\"\"

PARAMETER num_predict  {best['num_predict']}
PARAMETER num_ctx      {best['num_ctx']}
PARAMETER num_thread   {best['num_thread']}
PARAMETER temperature  0.3
PARAMETER top_p        0.9
PARAMETER top_k        20
PARAMETER repeat_penalty 1.15
PARAMETER stop "</s>"
PARAMETER stop "[INST]"
PARAMETER stop "\\n\\n"
PARAMETER stop "\\n3."
PARAMETER stop "\\n4."
"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(MODELFILE_PATH, "w") as f:
        f.write(content)
    print(f"\n  Modelfile written -> {MODELFILE_PATH}")


# ── Register + Validate ───────────────────────────────────────────────────────

def create_model():
    print(f"\n[3/4] Creating '{CUSTOM_MODEL}' in Ollama...")
    r = subprocess.run(
        ["ollama", "create", CUSTOM_MODEL, "-f", MODELFILE_PATH],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        print(f"  '{CUSTOM_MODEL}' created.")
    else:
        print(f"  [error] {r.stderr}")
        sys.exit(1)


def validate(baseline_latency: float, best_latency: float):
    print(f"\n[4/4] Final validation (3 runs avg)...")

    times = []
    llm = LocalLLM(model=CUSTOM_MODEL, temperature=0.3, max_tokens=150, timeout=120)
    for i in range(3):
        t0 = time.time()
        llm.generate(BENCHMARK_PROMPT)
        times.append(round(time.time() - t0, 2))
        print(f"  Run {i+1}: {times[-1]}s")

    final = round(sum(times) / len(times), 2)
    delta = round(baseline_latency - final, 1)
    pct   = round(delta / baseline_latency * 100, 1)

    print(f"\n  {'='*44}")
    print(f"  Baseline  ({BASELINE_MODEL}): {baseline_latency}s")
    print(f"  Grid best (search):    {best_latency}s")
    print(f"  Validated (3-run avg): {final}s")
    print(f"  Improvement: {'+' if delta > 0 else ''}{delta}s  ({pct}%)")
    print(f"  {'='*44}")

    if delta > 0:
        print(f"\n  Update your scripts:")
        print(f"    llm = LocalLLM(model='{CUSTOM_MODEL}', max_tokens=150, timeout=120)")
    else:
        print(f"\n  [note] No improvement found with param tuning alone.")
        print(f"  Consider switching to a smaller model:")
        print(f"    ollama pull phi3:mini")
        print(f"    ollama pull gemma2:2b")
        print(f"  Then rerun finetune.py after changing BASELINE_MODEL above.")


# ── Baseline timing ───────────────────────────────────────────────────────────

def time_baseline() -> float:
    print(f"  Timing baseline ({BASELINE_MODEL}, 2 runs)...", end=" ", flush=True)
    try:
        llm = LocalLLM(model=BASELINE_MODEL, temperature=0.3, max_tokens=300, timeout=180)
        times = []
        for _ in range(2):
            t0 = time.time()
            llm.generate(BENCHMARK_PROMPT)
            times.append(time.time() - t0)
        avg = round(sum(times) / len(times), 2)
        print(f"{avg}s")
        return avg
    except Exception as e:
        print(f"failed ({e})")
        return 30.0   # fallback


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Local LLM Optimizer  (CPU mode)")
    print("=" * 60)

    print("\n[1/4] Detecting hardware...")
    cpu = detect_cpu()

    baseline_latency = time_baseline()
    print(f"  Baseline latency: {baseline_latency}s")

    best = grid_search(cpu)

    write_modelfile(best)

    create_model()

    validate(baseline_latency, best["latency"])

    print("\nDone.")


if __name__ == "__main__":
    main()