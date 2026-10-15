"""
guard_stress_scoring.py - scores and reports for eval/guard_stress.py (kept separate so `build` needs no models).

Two stages:
  score        candidate checks per item, cached in guard_stress_scores.jsonl:
                 sim        cosine similarity, MiniLM (the RETIRED guard-v2 stage; kept only to document why it was removed)
                 overlap    share of the sentence's content words found in the chunk
                 specifics  share of the sentence's anchors (numbers, acronyms, names) found in the chunk
                 nli        FROZEN historical scores from the DeBERTa NLI model (guard v1), which has since been removed from
                            the codebase; new items get no nli score, and the NLI rows below only use the frozen values
                 llm        strict per-claim yes/no from an LLM (one call per claim; the first experiment)
  score-final  the SHIPPED LLM verifier (backend/app/citations/llm_verifier.py, batched claims per call) on the same
               items, cached in guard_stress_llmbatch.jsonl
  report       comparison table of the candidate checks (thresholds chosen on the real labelled sentences only)
  report-final the shipped layered guard (layer A = guard.screen_claim, layer B = cached verdicts) per mutation type
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.citations.guard import lexical_overlap, missing_anchors, screen_claim
from eval.guard_stress import CATEGORIES, STRESS_SCORES, STRESS_SET
from eval.label_validation import DATA, _read

LLMBATCH = os.path.join(DATA, "guard_stress_llmbatch.jsonl")

LLM_PROMPT = (
    "You are a strict fact-checker. Read the PASSAGE and the CLAIM.\n"
    "Does the passage itself state, or directly and unambiguously imply, the claim? Check every number, every name, "
    "every qualifier, and the direction of every comparison. If the claim adds anything the passage does not give, or "
    "contradicts it, the answer is NO.\n\nPASSAGE:\n{chunk}\n\nCLAIM:\n{claim}\n\n"
    'Reply with exactly one JSON object: {{"supported": true}} or {{"supported": false}}.')


def specifics_coverage(sentence: str, chunk: str) -> float:
    return missing_anchors(sentence, [chunk])[0]


def _llm_verdict(llm, chunk: str, claim: str) -> bool | None:
    try:
        out = llm.generate(LLM_PROMPT.format(chunk=chunk, claim=claim))
    except Exception:
        return None
    m = re.search(r'"supported"\s*:\s*(true|false)', out, re.I)
    return None if not m else m.group(1).lower() == "true"


def _cosine(embedder, a: str, b: str) -> float:
    import numpy as np
    v = np.asarray(embedder.embed_documents([a, b]), dtype="float64")
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return float(v[0] @ v[1])


def score_all(llm_per_category: int = 0) -> None:
    from backend.app.retrieval.embedder import get_embedder
    rows = _read(STRESS_SET)
    done = {r["item"]: r for r in _read(STRESS_SCORES)}
    todo = [r for r in rows if r["item"] not in done]
    print(f"{len(rows)} items, {len(done)} scored, {len(todo)} to score", flush=True)
    emb = get_embedder()
    llm = None
    if llm_per_category:
        from backend.app.llm.router import LLMRouter
        llm = LLMRouter()
    per_cat = defaultdict(int)
    for r in rows:
        if r["item"] in done and "llm" in done[r["item"]]:
            per_cat[r["category"]] += 1
    out = open(STRESS_SCORES, "a", encoding="utf-8")
    try:
        for k, r in enumerate(todo, 1):
            rec = {"item": r["item"], "category": r["category"], "label": r["label"],
                   "sim": _cosine(emb, r["chunk"], r["sentence"]),
                   "overlap": lexical_overlap(r["sentence"], r["chunk"]),
                   "specifics": specifics_coverage(r["sentence"], r["chunk"])}
            real = r["category"].startswith("real_")
            if llm is not None and (real or per_cat[r["category"]] < llm_per_category):
                v = _llm_verdict(llm, r["chunk"], r["sentence"])
                if v is not None:
                    rec["llm"] = v
                    per_cat[r["category"]] += 1
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if k % 20 == 0:
                print(f"  {k}/{len(todo)}", flush=True)
    finally:
        out.close()


def score_final(batch_size: int = 5) -> None:
    """Run the shipped batched LLM verifier on every item that has a per-claim LLM verdict (same sample)."""
    from backend.app.citations.llm_verifier import LLMEntailment
    from backend.app.llm.router import LLMRouter
    rows = {r["item"]: r for r in _read(STRESS_SET)}
    sample = [r["item"] for r in _read(STRESS_SCORES) if "llm" in r]
    done = {r["item"] for r in _read(LLMBATCH)}
    todo = [i for i in sample if i not in done]
    print(f"{len(sample)} items in the sample, {len(done)} already done, {len(todo)} to do", flush=True)
    llm = LLMRouter()
    verifier = LLMEntailment(llm)                      # exactly the class the pipeline uses
    out = open(LLMBATCH, "a", encoding="utf-8")
    calls = failures = 0
    try:
        for k in range(0, len(todo), batch_size):
            ids = todo[k:k + batch_size]
            calls += 1
            res = verifier.verify([(rows[i]["chunk"], rows[i]["sentence"]) for i in ids])
            if res is None:
                failures += 1
                continue
            for i, ok in zip(ids, res):
                out.write(json.dumps({"item": i, "supported": ok, "provider": llm.last_provider}) + "\n")
            out.flush()
            print(f"  {min(k + batch_size, len(todo))}/{len(todo)} (calls {calls}, unusable replies {failures})", flush=True)
    finally:
        out.close()
    print(f"done: {calls} calls, {failures} unusable replies")


# ── reports ───────────────────────────────────────────────────────────────────

def _pick_threshold(real_pos: list[float], max_false_drop: float = 0.20) -> float:
    """Highest threshold t (grid of 0.05) such that at most `max_false_drop` of real SUPPORTED sentences score < t."""
    best = 0.0
    for i in range(1, 20):
        t = i / 20
        if sum(1 for v in real_pos if v < t) / len(real_pos) <= max_false_drop:
            best = t
    return best


def report(out_path: str | None = None) -> dict:
    sc = _read(STRESS_SCORES)
    pos = [r for r in sc if r["category"] == "real_supported"]
    real_neg = [r for r in sc if r["category"] == "real_unsupported"]
    cats = [c for c in CATEGORIES if any(r["category"] == c for r in sc)]
    th = {k: _pick_threshold([r[k] for r in pos]) for k in ("nli", "sim", "overlap")}
    th["specifics"] = 1.0

    def keeps(r, name):
        if name == "llm":
            return r.get("llm", True)
        return r[name] >= th[name]

    methods = {
        "nli (guard v1, removed; frozen scores)": lambda r: keeps(r, "nli"),
        "cosine (retired v2 stage)": lambda r: keeps(r, "sim"),
        "overlap": lambda r: keeps(r, "overlap"),
        "cosine AND overlap (guard v2)": lambda r: keeps(r, "sim") and keeps(r, "overlap"),
        "specifics (all anchors found)": lambda r: keeps(r, "specifics"),
        "overlap AND specifics": lambda r: keeps(r, "overlap") and keeps(r, "specifics"),
        "nli AND overlap (frozen)": lambda r: keeps(r, "nli") and keeps(r, "overlap"),
        "nli AND overlap AND specifics (frozen)": lambda r: keeps(r, "nli") and keeps(r, "overlap") and keeps(r, "specifics"),
        "llm per-claim": lambda r: keeps(r, "llm"),
        "llm per-claim AND overlap": lambda r: keeps(r, "llm") and keeps(r, "overlap"),
    }
    has_llm = any("llm" in r for r in sc)
    res = {"thresholds_chosen_on_real_supported": th, "n_real_supported": len(pos), "n_real_unsupported": len(real_neg),
           "n_by_category": {c: sum(1 for r in sc if r["category"] == c) for c in cats}, "methods": {}}
    for name, fn in methods.items():
        if "llm" in name and not has_llm:
            continue
        use_pos = [r for r in pos if "llm" in r] if "llm" in name else pos
        use_cat = lambda c: [r for r in sc if r["category"] == c and ("llm" in r or "llm" not in name)]
        fd = sum(1 for r in use_pos if not fn(r)) / len(use_pos) if use_pos else None
        rec = {"false_drop_real_supported": None if fd is None else round(fd, 3),
               "caught_real_unsupported": f"{sum(1 for r in real_neg if not fn(r))}/{len(real_neg)}"}
        tot_c = tot_n = 0
        for c in cats:
            rs = use_cat(c)
            if rs:
                caught = sum(1 for r in rs if not fn(r))
                rec[c] = round(caught / len(rs), 3)
                tot_c += caught
                tot_n += len(rs)
        rec["synthetic_overall"] = round(tot_c / tot_n, 3) if tot_n else None
        res["methods"][name] = rec
    _print_table(res, cats, f"candidate checks; thresholds chosen on the real supported sentences (max 20% false drops): {th}")
    p = out_path or os.path.join(DATA, "guard_stress_report.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


def report_final(out_path: str | None = None) -> dict:
    """The shipped layered guard: layer A = guard.screen_claim on the item text, layer B = cached verdicts."""
    from backend.app import config
    sc = {r["item"]: r for r in _read(STRESS_SCORES)}
    rows = {r["item"]: r for r in _read(STRESS_SET)}
    batched = {r["item"]: r["supported"] for r in _read(LLMBATCH)}
    cats = [c for c in CATEGORIES if any(r["category"] == c for r in sc.values())]

    def layer_a(item):
        r = rows[item]
        return screen_claim(r["sentence"], [r["chunk"]])["ok"]

    variants = {
        "A only (overlap floor + specifics)": (lambda i: layer_a(i), False),
        "A + LLM batched (shipped default)": (lambda i: layer_a(i) and batched[i], True),
        "LLM batched only (for reference)": (lambda i: batched[i], True),
    }
    res = {"layer_a": {"overlap_floor": config.GUARD_OVERLAP_THRESHOLD, "specifics_min": config.GUARD_SPECIFICS_MIN},
           "sample_size_llm": len(batched), "methods": {}}
    for name, (fn, needs_llm) in variants.items():
        def sel(cat):
            return [i for i, r in sc.items() if r["category"] == cat and (not needs_llm or i in batched)]
        pos, neg = sel("real_supported"), sel("real_unsupported")
        rec = {"false_drop_real_supported": round(sum(1 for i in pos if not fn(i)) / len(pos), 3),
               "caught_real_unsupported": f"{sum(1 for i in neg if not fn(i))}/{len(neg)}"}
        tc = tn = 0
        for c in cats:
            ids = sel(c)
            if ids:
                caught = sum(1 for i in ids if not fn(i))
                rec[c] = round(caught / len(ids), 3)
                tc += caught
                tn += len(ids)
        rec["synthetic_overall"] = round(tc / tn, 3) if tn else None
        rec["n_real_supported"] = len(pos)
        res["methods"][name] = rec
    _print_table(res, cats, "shipped guard, evaluated on the same items (LLM variants use the batched-verifier sample)")
    p = out_path or os.path.join(DATA, "guard_final_report.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    return res


def _print_table(res: dict, cats: list[str], title: str) -> None:
    print(title)
    print(" | ".join(["method", "FD%", "real-neg"] + [c[:9] for c in cats] + ["synth-all"]))
    for name, rec in res["methods"].items():
        row = [name, f"{100 * rec['false_drop_real_supported']:.0f}" if rec["false_drop_real_supported"] is not None else "-",
               rec["caught_real_unsupported"]] + [f"{100 * rec[c]:.0f}" if c in rec else "-" for c in cats] + \
              [f"{100 * rec['synthetic_overall']:.0f}" if rec["synthetic_overall"] is not None else "-"]
        print(" | ".join(row))
