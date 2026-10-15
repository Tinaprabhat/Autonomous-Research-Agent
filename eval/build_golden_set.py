"""
build_golden_set.py — builds the golden set from the ingested corpus, with automatic verification.

    python -m eval.build_golden_set --local [--answerable 80] [--unanswerable 20] [--allow-partial]

NO human review is involved, so every case must survive machine checks before it is kept:

Answerable (gold = the source chunk, so each case is a cited ground truth):
  1. sample a prose chunk from a main paper section (<=2 per paper, so coverage is spread),
  2. an LLM writes a self-contained question + a short answer using ONLY that passage,
  3. reject questions that lean on context ("this paper", "the passage", ...) or duplicate another,
  4. a DIFFERENT LLM cross-checks: answerable from the passage alone? answer correct? question standalone?
Unanswerable:
  1. an LLM proposes on-topic, corpus-specific questions the papers likely do not cover,
  2. the hybrid retriever fetches top passages for each and the checker LLM judges whether ANY of them
     answers it; only questions where none do are kept (this is what prevents "capital of France" leaks).

Free-tier safe: generation and verification use two different Groq models (independent cross-check and
separate daily token quotas). Every accepted case and every tried chunk is checkpointed to
data/eval_runs/golden_partial.jsonl, so the build resumes where it stopped. If a model's DAILY quota is
exhausted the build stops cleanly (exit code 3); re-run the same command later. --allow-partial writes
the golden set from whatever has been verified so far.

Cases are stored with verification="auto" (not "reviewed"), the first 20 answerable ones flagged
regression=true. Limitations: machine-checked ground truth is weaker than human labels, and other
chunks besides the gold one may also answer a question (chunk recall is a lower bound).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval import golden

MAIN_SECTIONS = {"Introduction", "Related Work", "Background", "Method", "Experiments", "Results",
                 "Discussion", "Conclusion", "Abstract"}
CONTEXT_LEAKS = re.compile(
    r"\b(this|the|these) (paper|passage|text|study|work|article|authors?|excerpt|section|figure|table)\b|"
    r"\b(the|this) proposed\b|\bin (this|the) (paper|study|work)\b|\baccording to the\b|"
    r"\b(described|referenced|mentioned|aforementioned|above|following|given|presented|considered) "
    r"(paper|work|method|model|approach|paradigm|setting|framework|dataset|study|task|experiment)s?\b|"
    r"\b(the|this|these) (reference|citation)s?\b|\breference \[?\d|"
    r"\[\d+(?:\s*[,;–-]\s*\d+)*\]|"                       # bracketed citation markers like [31, 27]
    r"\b(mentioned|discussed|stated|noted|reported|described|shown|listed) (in|by) (the|this)\b|"
    r"\b(concurrent|previous|prior|related|earlier|aforementioned) (works?|studies|papers|approaches)\b", re.I)


def question_ok(q: str) -> bool:
    """Deterministic standalone-question gate (the LLM cross-check alone proved too lenient)."""
    body = q.strip()
    if not body.endswith("?") or len(body.split()) < 6:
        return False
    if "?" in body[:-1] or "�" in body:       # garbled glyphs from PDF extraction (e.g. "Estrogen??")
        return False
    return not CONTEXT_LEAKS.search(body)

Q_PROMPT = """Below is a passage from an academic paper titled "{title}".
Write ONE specific, self-contained factual question that this passage answers, plus a short answer.
Rules:
- Name the method / model / dataset / concept explicitly so the question is unambiguous WITHOUT the passage.
- Do NOT write "this paper", "the passage", "the authors", "the study", "the proposed method".
- The answer must be 1-2 sentences and use ONLY facts stated in the passage.
Return JSON only: {{"question": "...", "answer": "..."}}

Passage:
{passage}"""

CHECK_PROMPT = """Passage:
{passage}

Question: {question}
Proposed answer: {answer}

Judge strictly using ONLY the passage. Return JSON only:
{{"answerable_from_passage": true/false, "answer_correct_and_supported": true/false,
  "standalone": true/false}}

"standalone" is true ONLY if a reader who has never seen the passage could tell exactly what the question
is about (it names the specific method, model, dataset or concept) and would not need to ask "which one?"."""

UNANS_PROMPT = """Titles of papers in a research corpus:
{titles}

Write {n} realistic, SPECIFIC questions a researcher might ask on these topics whose answers are very likely
NOT stated in any of these papers (e.g. a particular benchmark number for a method they do not evaluate, a
follow-up result, an experiment they never ran, a comparison against a method they do not mention).
They must be about the research topics (not general trivia). One per line, no numbering."""

ANSWERABLE_CHECK = """Passages:
{passages}

Question: {question}

Does ANY passage contain the information needed to answer this question? Return JSON only:
{{"answerable": true/false}}"""

PARTIAL = "data/eval_runs/golden_partial.jsonl"


class BudgetExhausted(RuntimeError):
    """A model's DAILY quota is used up; retrying today is pointless."""


def jparse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no json")
    return json.loads(m.group(0))


def call(llm, prompt: str, tries: int = 6) -> str:
    """Paced LLM call: waits out per-minute limits, raises BudgetExhausted on daily quotas."""
    from backend.app.llm.groq_llm import LLMError, LLMRateLimitError
    for i in range(tries):
        try:
            out = llm.generate(prompt)
            time.sleep(1.5)
            return out
        except LLMRateLimitError as e:
            if e.daily and (e.retry_after is None or e.retry_after > 180):
                raise BudgetExhausted(f"{getattr(llm, 'model', '?')}: {e}") from e
            if i == tries - 1:
                raise
            time.sleep(min((e.retry_after or 10) + 1, 75))
        except LLMError:
            if i == tries - 1:
                raise
            time.sleep(3 + 3 * i)          # transient network / API error
    raise RuntimeError("unreachable")


class Checkpoint:
    """Append-only JSONL log of accepted cases and tried chunks (fsync'd per record)."""

    def __init__(self, path: str = PARTIAL):
        self.path = path
        self.cases: list[golden.GoldenCase] = []
        self.tried: set[str] = set()
        if path != os.devnull and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue            # torn last line after a crash
                    if r.get("kind") == "tried":
                        self.tried.add(r["chunk_id"])
                    elif r.get("kind") == "case":
                        case = golden.GoldenCase(**r["case"])
                        # re-apply the current question gate to cases accepted under older, looser rules
                        if not case.answerable or question_ok(case.question):
                            self.cases.append(case)

    def _append(self, rec: dict) -> None:
        if self.path == os.devnull:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def mark_tried(self, chunk_id: str) -> None:
        self.tried.add(chunk_id)
        self._append({"kind": "tried", "chunk_id": chunk_id})

    def add_case(self, case: golden.GoldenCase) -> None:
        self.cases.append(case)
        self._append({"kind": "case", "case": case.__dict__})


def is_prose(c: dict) -> bool:
    t = c["text"]
    letters = sum(ch.isalpha() for ch in t)
    return (len(t.split()) >= 45 and letters / max(len(t), 1) > 0.72 and t.count("?") < 3
            and c.get("section") in MAIN_SECTIONS and not t.lstrip().startswith(("Figure", "Table")))


def jaccard(a: str, b: str) -> float:
    A, B = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(A & B) / max(len(A | B), 1)


def build_answerable(gen_llm, check_llm, chunks, target: int, log,
                     ck: Checkpoint | None = None) -> list[golden.GoldenCase]:
    ck = ck or Checkpoint(os.devnull)
    cases = [c for c in ck.cases if c.answerable]
    per_paper: dict[str, int] = {}
    for c in cases:
        for a in c.gold_arxiv_ids:
            per_paper[a] = per_paper.get(a, 0) + 1
    stats = {"tried": 0, "ctx_leak": 0, "dup": 0, "llm": 0, "err": 0, "resumed": len(cases)}
    for c in chunks:
        if len(cases) >= target:
            break
        if c["chunk_id"] in ck.tried or per_paper.get(c["arxiv_id"], 0) >= 2:
            continue
        stats["tried"] += 1
        try:
            j = jparse(call(gen_llm, Q_PROMPT.format(title=c.get("title", ""), passage=c["text"])))
            q, ans = j["question"].strip(), j["answer"].strip()
            if not question_ok(q):
                stats["ctx_leak"] += 1
                ck.mark_tried(c["chunk_id"])
                continue
            if any(jaccard(q, x.question) > 0.7 for x in cases):
                stats["dup"] += 1
                ck.mark_tried(c["chunk_id"])
                continue
            v = jparse(call(check_llm, CHECK_PROMPT.format(passage=c["text"], question=q, answer=ans)))
            if not (v.get("answerable_from_passage") is True and v.get("answer_correct_and_supported") is True
                    and v.get("standalone") is True):
                stats["llm"] += 1
                ck.mark_tried(c["chunk_id"])
                continue
        except BudgetExhausted:
            log(f"answerable stats: {stats}")
            raise
        except Exception as e:           # one bad sample must never kill the run
            stats["err"] += 1
            log(f"  [skip] {type(e).__name__}: {str(e)[:100]}")
            continue                      # not marked tried: a transient failure may be retried on resume
        per_paper[c["arxiv_id"]] = per_paper.get(c["arxiv_id"], 0) + 1
        case = golden.GoldenCase(
            id=f"a{len(cases) + 1:03d}", question=q, answerable=True,
            gold_chunk_ids=[c["chunk_id"]], gold_arxiv_ids=[c["arxiv_id"]], reference_answer=ans,
            verification="auto", regression=len(cases) < 20,
            source_page=c.get("page"), source_section=c.get("section"))
        cases.append(case)
        ck.mark_tried(c["chunk_id"])
        ck.add_case(case)
        log(f"answerable {len(cases)}/{target}  (tried {stats['tried']})")
    log(f"answerable stats: {stats}")
    return cases


def build_unanswerable(gen_llm, check_llm, retriever, titles: list[str], target: int, log,
                       ck: Checkpoint | None = None) -> list[golden.GoldenCase]:
    ck = ck or Checkpoint(os.devnull)
    cases = [c for c in ck.cases if not c.answerable]
    seen = [c.question for c in cases]
    rounds = 0
    title_block = "\n".join(f"- {t}" for t in titles[:70])
    while len(cases) < target and rounds < 8:
        rounds += 1
        try:
            out = call(gen_llm, UNANS_PROMPT.format(titles=title_block, n=20))
        except BudgetExhausted:
            raise
        except Exception as e:
            log(f"  [skip round] {type(e).__name__}")
            continue
        for line in out.splitlines():
            q = re.sub(r"^[\-\d\.\)\s\*]+", "", line).strip()
            if not q.endswith("?") or len(q.split()) < 7 or any(jaccard(q, s_) > 0.7 for s_ in seen):
                continue
            seen.append(q)
            try:
                docs = retriever.search(q, k=6, rewrite=False)
                passages = "\n\n".join(f"[{i}] {d['text']}" for i, d in enumerate(docs, 1))
                v = jparse(call(check_llm, ANSWERABLE_CHECK.format(passages=passages, question=q)))
            except BudgetExhausted:
                raise
            except Exception as e:
                log(f"  [skip] {type(e).__name__}")
                continue
            if v.get("answerable") is False:
                case = golden.GoldenCase(id=f"u{len(cases) + 1:03d}", question=q,
                                         answerable=False, verification="auto")
                cases.append(case)
                ck.add_case(case)
                log(f"unanswerable {len(cases)}/{target}")
            if len(cases) >= target:
                break
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerable", type=int, default=80)
    ap.add_argument("--unanswerable", type=int, default=20)
    ap.add_argument("--out", default=golden.DEFAULT_PATH)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true", help="overwrite an existing golden set")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write the golden set even if fewer cases than requested were verified")
    ap.add_argument("--fresh", action="store_true", help="ignore/delete the resume checkpoint")
    ap.add_argument("--local", action="store_true")
    a = ap.parse_args()
    if a.local:
        os.environ["QDRANT_FORCE_LOCAL"] = "1"
    if os.path.exists(a.out) and not a.force:
        sys.exit(f"{a.out} exists — pass --force to overwrite.")
    if a.fresh and os.path.exists(PARTIAL):
        os.remove(PARTIAL)

    from backend.app import config
    from backend.app.llm.groq_llm import GroqLLM
    from backend.app.retrieval.embedder import get_embedder
    from backend.app.retrieval.hybrid_retriever import HybridRetriever
    from backend.app.storage.qdrant_store import QdrantStore

    log = lambda m: print(m, flush=True)
    gen_llm = GroqLLM(model=config._env("GOLDEN_GEN_MODEL", default="qwen/qwen3.8-27b"),
                      max_tokens=600, timeout=45.0)
    check_llm = GroqLLM(model=config._env("GOLDEN_CHECK_MODEL", default="openai/gpt-oss-20b"),
                        max_tokens=400, timeout=45.0)
    log(f"generator: {gen_llm.model} | checker: {check_llm.model}")
    store = QdrantStore()
    all_chunks = store.scroll_all()
    chunks = [c for c in all_chunks if is_prose(c)]
    random.Random(a.seed).shuffle(chunks)
    ck = Checkpoint()
    log(f"{len(chunks)} prose chunks eligible of {len(all_chunks)}; resuming with {len(ck.cases)} cases, "
        f"{len(ck.tried)} chunks already tried")

    exhausted = None
    try:
        answerable = build_answerable(gen_llm, check_llm, chunks, a.answerable, log, ck)
        titles = sorted({c["title"] for c in all_chunks if c.get("title")})
        random.Random(a.seed).shuffle(titles)
        unanswerable = build_unanswerable(gen_llm, check_llm, HybridRetriever(store, get_embedder(), all_chunks),
                                          titles, a.unanswerable, log, ck)
    except BudgetExhausted as e:
        exhausted = str(e)[:300]
        answerable = [c for c in ck.cases if c.answerable]
        unanswerable = [c for c in ck.cases if not c.answerable]
        log(f"\nDAILY QUOTA EXHAUSTED: {exhausted}")

    for i, c in enumerate(answerable, start=1):      # stable, gap-free ids after any filtering
        c.id, c.regression = f"a{i:03d}", i <= 20
    for i, c in enumerate(unanswerable, start=1):
        c.id = f"u{i:03d}"
    cases = answerable + unanswerable
    complete = len(answerable) >= a.answerable and len(unanswerable) >= a.unanswerable
    log(f"verified so far: {len(answerable)}/{a.answerable} answerable, "
        f"{len(unanswerable)}/{a.unanswerable} unanswerable")
    if complete or (a.allow_partial and cases):
        golden.save(cases, a.out)
        log(f"wrote {len(cases)} cases -> {a.out}" + ("" if complete else "  (PARTIAL)"))
    if exhausted:
        log("Re-run the same command later: it resumes from data/eval_runs/golden_partial.jsonl.")
        sys.exit(3)
    if not complete:
        log("WARNING: fewer cases than requested survived verification.")
        sys.exit(0 if a.allow_partial else 4)


if __name__ == "__main__":
    main()
