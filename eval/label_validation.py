"""
label_validation.py - validate the LLM judge and the NLI citation threshold against hand labels.

    python -m eval.label_validation build  [--n 30] [--local]     # run 30 answerable golden cases, save blind label set
    python -m eval.label_validation show                          # print the blind items (no scores) for labelling
    python -m eval.label_validation report [--labels eval/data/labels_human.jsonl ...]

What is labelled (per answer, on the UNGUARDED model answer, so wrong sentences are present to be found):
  * sentence label  "supported": a sentence that carries a valid [N] marker is supported if at least one of the
                    passages it cites entails it (the same rule the NLI guard applies).
  * answer label    "faithful":  every factual claim in the whole answer is supported by the retrieved passages
                    (the question DeepEval's Faithfulness metric is meant to answer).
The label set (eval/data/label_set.jsonl) deliberately contains NO model scores, so labelling is blind.
Scores live in eval/data/label_scores.jsonl: the NLI entailment per sentence and the DeepEval faithfulness
per answer (judge: Groq gpt-oss-20b). `report` joins labels with scores and computes:
  * NLI threshold sweep: precision / recall / F1 / accuracy per threshold, the guard's 0.5 operating point, best F1, AUC
  * judge agreement: accuracy, Cohen's kappa, confusion matrix vs the answer labels, Spearman vs share of supported sentences
  * label agreement between labelers (e.g. human vs Claude) when more than one label file is given
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DATA = os.path.join(os.path.dirname(__file__), "data")
LABEL_SET = os.path.join(DATA, "label_set.jsonl")
SCORES = os.path.join(DATA, "label_scores.jsonl")
CKPT = "data/eval_runs/generation_ckpt.jsonl"
RAW_CFG = "hybrid+rerank (no guard)"
SEED = 13
THRESHOLDS = [round(x, 2) for x in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]]


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _append(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# -- build ---------------------------------------------------------------------

def split_cited(answer: str, n_chunks: int):
    """[(sentence_with_markers, hypothesis_without_markers, refs)] using the guard's own sentence splitter."""
    import re
    from backend.app.citations.guard import _CITATION_RE, _split_sentences
    out = []
    for sent in _split_sentences(answer):
        refs = sorted({int(m) for m in _CITATION_RE.findall(sent) if 1 <= int(m) <= n_chunks})
        hyp = re.sub(r"\s+", " ", _CITATION_RE.sub("", sent)).strip()
        out.append((sent, hyp, refs))
    return out


def build(n: int = 30, local: bool = True):
    if local:
        os.environ["QDRANT_FORCE_LOCAL"] = "1"
    from eval import golden, runner as R
    from eval.generation import JudgeBudgetExhausted, score_case
    from eval.judge import GroqJudge
    from backend.app.services import build_pipeline

    cases = [c for c in golden.load(only_reviewed=True) if c.answerable]
    ckpt = {r["case_id"]: r for r in _read(CKPT) if r.get("answerable")}
    rng = random.Random(SEED)
    fresh_pool = sorted(c.id for c in cases if c.id not in ckpt)
    chosen = sorted(ckpt) + sorted(rng.sample(fresh_pool, max(n - len(ckpt), 0)))
    chosen = chosen[:n]
    by_id = {c.id: c for c in cases}
    done = {r["case_id"] for r in _read(SCORES)}
    todo = [cid for cid in chosen if cid not in done]
    print(f"{len(chosen)} cases selected, {len(done)} already built, {len(todo)} to build")
    if not todo:
        return

    pipeline, _ = build_pipeline(use_rewriter=False)
    judge = GroqJudge()
    faith = R.deepeval_metrics(judge)["faithfulness"]
    from deepeval.test_case import LLMTestCase
    for cid in todo:
        c = by_id[cid]
        if cid in ckpt:
            ctx, raw = ckpt[cid]["context"], ckpt[cid]["raw_answer"]
            judge_score = (ckpt[cid]["scores"].get(RAW_CFG) or {}).get("faithfulness")
            src = "checkpoint"
        else:
            chunks = R.retrieve_for_config(pipeline, "hybrid+rerank", c.question, 5)
            ctx = [d["text"] for d in chunks]
            raw = R.raw_answer(pipeline, c.question, chunks)
            judge_score, src = None, "fresh"
        if judge_score is None:
            tc = LLMTestCase(input=c.question, actual_output=raw, retrieval_context=ctx)
            try:
                judge_score = score_case({"faithfulness": faith}, tc, ["faithfulness"]).get("faithfulness")
            except JudgeBudgetExhausted as e:
                print(f"STOPPED: judge daily quota exhausted ({e}); re-run to resume")
                return
        sents = split_cited(raw, len(ctx))
        pairs, owner = [], []
        for si, (_, hyp, refs) in enumerate(sents):
            for r in refs:
                pairs.append((ctx[r - 1], hyp))
                owner.append((si, r))
        probs = pipeline.verifier.entailment_batch(pairs) if pairs else []
        per_sent = {}
        for (si, r), p in zip(owner, probs):
            per_sent.setdefault(si, {})[r] = round(float(p), 4)
        items, scores = [], []
        for si, (sent, hyp, refs) in enumerate(sents):
            if not refs:
                continue
            iid = f"{cid}#{si}"
            items.append({"id": iid, "sentence": hyp, "refs": refs})
            scores.append({"id": iid, "nli_by_ref": per_sent[si], "nli": max(per_sent[si].values())})
        _append(LABEL_SET, {"case_id": cid, "question": c.question, "context": ctx, "answer": raw, "items": items})
        _append(SCORES, {"case_id": cid, "source": src, "judge_faithfulness": judge_score,
                         "n_sentences": len(sents), "n_cited": len(items), "sentences": scores})
        print(f"  built {cid} ({src}): {len(items)} cited sentences, judge={judge_score}")


def show():
    """Print the blind label set (no scores)."""
    for r in _read(LABEL_SET):
        print("=" * 90)
        print(f"CASE {r['case_id']}  Q: {r['question']}")
        for i, c in enumerate(r["context"], 1):
            print(f"  [{i}] {c}")
        print("  ANSWER:", r["answer"])
        for it in r["items"]:
            print(f"  ITEM {it['id']} refs={it['refs']}: {it['sentence']}")


# -- statistics ------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(centre - half, 0), 3), round(min(centre + half, 1), 3))


def confusion(pred, truth):
    tp = sum(1 for p, t in zip(pred, truth) if p and t)
    fp = sum(1 for p, t in zip(pred, truth) if p and not t)
    fn = sum(1 for p, t in zip(pred, truth) if not p and t)
    tn = sum(1 for p, t in zip(pred, truth) if not p and not t)
    return tp, fp, fn, tn


def metrics_from(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp and prec == prec and rec == rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": round((tp + tn) / n, 3) if n else float("nan"),
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def auc(scores, truth):
    """P(score of a random supported sentence > score of a random unsupported one); ties count half."""
    pos = [s for s, t in zip(scores, truth) if t]
    neg = [s for s, t in zip(scores, truth) if not t]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 3) if den else float("nan")


def load_labels(path):
    sent, ans, labeler = {}, {}, None
    for r in _read(path):
        labeler = r.get("labeler", labeler)
        if r.get("exclude"):                       # e.g. a sentence-splitter fragment that is not a claim
            continue
        if r["type"] == "sentence":
            sent[r["id"]] = bool(r["supported"])
        elif r["type"] == "answer":
            ans[r["id"]] = bool(r["faithful"])
    return sent, ans, labeler


def analyse(labels_path: str, judge_pass: float = 0.5, nli_default: float = 0.5) -> dict:
    sent_lab, ans_lab, labeler = load_labels(labels_path)
    scores = _read(SCORES)
    nli = {s["id"]: s["nli"] for r in scores for s in r["sentences"]}
    judge = {r["case_id"]: r["judge_faithfulness"] for r in scores}

    ids = [i for i in sent_lab if i in nli]
    y = [sent_lab[i] for i in ids]
    x = [nli[i] for i in ids]
    sweep = []
    for t in THRESHOLDS:
        m = metrics_from(*confusion([v >= t for v in x], y))
        sweep.append({"threshold": t, **m})
    at = next(s for s in sweep if s["threshold"] == nli_default)
    best = max(sweep, key=lambda s: (s["f1"], s["accuracy"]))
    nli_res = {
        "n_sentences": len(ids), "n_supported": sum(y), "n_unsupported": len(y) - sum(y),
        "auc": auc(x, y), "at_guard_threshold": at, "best_f1": best, "sweep": sweep,
        "accuracy_ci95_at_guard_threshold": wilson(at["tp"] + at["tn"], len(ids)),
        "false_drops_at_guard_threshold": at["fn"], "false_keeps_at_guard_threshold": at["fp"],
    }

    cids = [c for c in ans_lab if judge.get(c) is not None]
    ay = [ans_lab[c] for c in cids]
    js = [float(judge[c]) for c in cids]
    jp = [v >= judge_pass for v in js]
    tp, fp, fn, tn = confusion(jp, ay)
    share = {}
    for c in cids:
        labs = [sent_lab[s["id"]] for r in scores if r["case_id"] == c for s in r["sentences"] if s["id"] in sent_lab]
        share[c] = sum(labs) / len(labs) if labs else None
    pairs = [(float(judge[c]), share[c]) for c in cids if share[c] is not None]
    judge_res = {
        "n_answers": len(cids), "n_faithful_by_label": sum(ay), "judge_pass_threshold": judge_pass,
        "agreement": round(sum(1 for a, b in zip(jp, ay) if a == b) / len(ay), 3) if ay else float("nan"),
        "agreement_ci95": wilson(sum(1 for a, b in zip(jp, ay) if a == b), len(ay)),
        "kappa": cohen_kappa(jp, ay),
        "confusion_judge_pass_vs_label_faithful": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "judge_misses_unfaithful": fp, "judge_flags_faithful": fn,
        "spearman_judge_vs_supported_share": spearman([p[0] for p in pairs], [p[1] for p in pairs]),
        "mean_judge_score": round(sum(js) / len(js), 3) if js else float("nan"),
        "label_faithful_rate": round(sum(ay) / len(ay), 3) if ay else float("nan"),
        "disagreements": [{"case_id": c, "judge": judge[c], "label_faithful": ans_lab[c]}
                          for c, p in zip(cids, jp) if p != ans_lab[c]],
    }
    return {"labeler": labeler, "labels_file": labels_path, "nli": nli_res, "judge": judge_res}


def label_agreement(path_a, path_b):
    sa, aa, la = load_labels(path_a)
    sb, ab, lb = load_labels(path_b)
    ks = [k for k in sa if k in sb]
    ka = [k for k in aa if k in ab]
    return {"labelers": [la, lb],
            "sentence": {"n": len(ks), "agreement": round(sum(sa[k] == sb[k] for k in ks) / len(ks), 3) if ks else None,
                         "kappa": cohen_kappa([sa[k] for k in ks], [sb[k] for k in ks])},
            "answer": {"n": len(ka), "agreement": round(sum(aa[k] == ab[k] for k in ka) / len(ka), 3) if ka else None,
                       "kappa": cohen_kappa([aa[k] for k in ka], [ab[k] for k in ka])}}


def report(label_files, out=None):
    res = {"analyses": [analyse(p) for p in label_files]}
    if len(label_files) >= 2:
        res["label_agreement"] = label_agreement(label_files[0], label_files[1])
    text = json.dumps(res, indent=2)
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--n", type=int, default=30); b.add_argument("--local", action="store_true")
    sub.add_parser("show")
    r = sub.add_parser("report"); r.add_argument("--labels", nargs="+", default=[os.path.join(DATA, "labels_claude.jsonl")])
    r.add_argument("--out", default=os.path.join(DATA, "validation_report.json"))
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.n, a.local)
    elif a.cmd == "show":
        show()
    else:
        report(a.labels, a.out)


if __name__ == "__main__":
    main()
