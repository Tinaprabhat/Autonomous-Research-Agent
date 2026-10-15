# Autonomous Research Agent (v2) — Project Report

A citation-grounded question-answering system over a corpus of 99 arXiv research papers. The model must put a `[N]`
citation on every claim; a layered citation guard then screens each cited sentence (matching numbers and names, word
overlap, and an entailment check by an LLM verifier) and removes or tags what it cannot support. There is no fallback
verifier: if the LLM check cannot run, the answer carries a visible warning.
The whole thing runs on a $0 stack (Groq free tier, Qdrant, CPU-only containers).

**The headline result is the error rate, and the guard has not been shown to lower it.** In a human-labelled sample of 30
answers, 5 raw (unguarded) answers (16.7%; 95% CI 7–34%) contained a claim the cited passages do not support. After the
shipped guard, 3 or 4 of the 25 answers returned without a "not sure" tag (12–16% depending on the run; CIs 4–35%) still did,
and 4 of 25 correct answers (16%) were wrongly tagged. Answer rate 83% (66–93%), false-refusal rate 3% (1–17%). The "not
sure" tag does not track correctness in this sample (every tagged answer was clean; none of the untagged errors was flagged),
so there is no calibrated confidence label (§11.7). The guard's own verifier is not deterministic across identical runs
(§11.6). With n = 30 none of this can distinguish "no effect" from a modest one. Treat the system as one that catches
mechanical errors (wrong numbers, names, citations) and sometimes makes an unsupported claim visible, not as one that
guarantees faithfulness.

Numbers in this report come from the repo's own files and runs (corpus files, `eval/reports/latest.json`,
`docs/case_study.md`, source code). Where a figure is an estimate rather than a measurement, it says so.

---

## 1. At a glance

| Item | Value |
|---|---|
| Domain | Machine learning / NLP / information retrieval research (meta-learning, RAG, dense retrieval, in-context learning, LoRA, hallucination detection, transformers, prompt injection) |
| Source | arXiv (public API, relevance-sorted queries) |
| Papers in manifest | 100 harvested |
| Papers ingested | **99** (1 genuine arXiv 404, not downloadable) |
| PDF files | 99 (about 227–237 MB) |
| Chunks (vectors) | **19,116** |
| Chunk size | 500-character cap; mean 309, median 330 chars |
| Chunk overlap | 70 characters (applied when a long semantic group is split) |
| RAG type | Hybrid (dense + BM25) with Reciprocal Rank Fusion, cross-encoder reranking, layered citation verification |
| Embedding model | `all-MiniLM-L6-v2` (384-d) |
| Vector DB | Qdrant (cosine, HNSW m=16 / ef_construct=100, 5 payload indexes) |
| Generator | Groq `openai/gpt-oss-120b` with fallback chain (see §8) |
| Golden evaluation set | 100 questions (80 answerable, 20 unanswerable) |
| Tests | **443 offline tests pass** (+ 12 slow classifier tests); 12/12 live end-to-end smoke checks (earlier run); 16 failure-injection tests |
| Auth / limits | API-key auth, per-user per-minute limit and daily quota (§11.1) |
| Backup | Qdrant Cloud snapshot (99.6 MB) + restore drill PASSED, 19,116/19,116 points identical (§11.3) |
| **Error / answer / false-refusal rates (post-guard, n = 30, Wilson 95% CI)** | **error among answered 12–16% across two identical runs (CIs ≈ 4–35%) vs 16.7% (7.3–33.6%) before; answer rate 83.3% (66.4–92.7%); false-refusal 3.3% (0.6–16.7%); correct answers wrongly tagged 16.0% (6.4–34.7%) (§11.6)** |
| Label calibration / relevance / completeness | confidence tag is uninformative (all 4 "not sure" answers were clean; guard-score AUROC 0.49); completeness 0.89 (0.77–0.99) and relevance 0.93 (0.83–1.00) by an LLM judge whose controls pass but that no human has checked (§11.7) |
| Generation metrics tool | DeepEval dropped (every answer scored 1.0); **RAGAS** now, all 30 cases, one judge: faithfulness 0.937 (0.862–0.995), hallucination 0.063 (0.005–0.138), completeness 0.783 (0.650–0.900), correctness 0.532 (0.372–0.708, an underestimate). RAGAS flags 2 of 5 human-unfaithful answers and 3 of 25 faithful ones (AUROC 0.63, CI 0.42–0.87) (§11.8) |
| Retrieval quality | hybrid + rerank: chunk recall@5 **0.912**, MRR **0.856**, paper recall@5 **1.000** |
| Unanswerable handling | 19/20 (95%) correctly refused |
| Median latency (CPU laptop) | about 8.6 s per query (worst 15 s) |
| Cost | $0 (free tiers only) |

---

## 2. Corpus

### 2.1 Source and selection
* **Source:** arXiv, through the official `arxiv` Python client, rate-limited to at least 3 seconds between requests
  with a single connection (arXiv API terms).
* **Harvest method:** 8 topical search queries, restricted to the categories `cs.LG`, `cs.CL`, `cs.IR`, `cs.AI`
  (the secondary-category list means some papers have a different *primary* category), sorted by relevance, de-duplicated
  by base arXiv id, capped at 100 papers:
  1. meta-learning few-shot learning
  2. retrieval-augmented generation
  3. dense passage retrieval
  4. in-context learning large language models
  5. parameter-efficient fine-tuning LoRA
  6. hallucination detection language models
  7. transformer self-attention
  8. prompt injection attacks language models
* **Result:** 100 papers harvested, 99 ingested. The one failure is a real arXiv 404.

### 2.2 Corpus statistics (computed from `data/manifest.json` and `data/processed_chunks/`)

| Statistic | Value |
|---|---|
| Papers in manifest | 100 |
| Papers with PDFs and chunks | 99 |
| Total chunks | 19,116 |
| Chunks per paper | min 57, mean 193.1, max 756 |
| Publication years | 2014 – 2026 |
| Paper versions | v1: 58, v2: 20, v3: 12, v4: 7, v5: 1, v6: 2 |
| Raw PDF size on disk | about 227–237 MB |
| Parsed chunk JSON on disk | about 15 MB |
| Local Qdrant store on disk | about 87 MB |

**Primary arXiv category (100 papers):** cs.CL 40, cs.LG 24, cs.IR 13, cs.CV 9, cs.CR 8, cs.AI 3, stat.ML 1,
cs.SD 1, cs.DB 1.

**Publication year (100 papers):** 2025: 29, 2024: 25, 2026: 13, 2023: 8, 2020: 7, 2022: 7, 2019: 4, 2018: 3, 2021: 3,
2014: 1 (the remaining years in 2015–2017 have none).

**Section labels of the 19,116 chunks (top):** Appendix 5,215 · Introduction 1,938 · Related Work 1,512 · Method 1,153 ·
Experiments 753 · Conclusion 667 · Discussion 665 · Acknowledgments 438 · Background 367 · Abstract 366 · Unknown 217 ·
Results 173. `Appendix` (about 27%) is a catch-all for back matter. See §12 for the caveat.

---

## 3. Ingestion pipeline

```
arXiv API ─▶ harvest metadata ─▶ SQLite job queue ─▶ download PDF ─▶ parse (PyMuPDF) ─▶ chunk ─▶ embed ─▶ Qdrant upsert
                                  pending → downloaded → parsed → chunked → embedded   (failed after ≤3 retries)
```

* **Job queue (`ingestion/job_queue.py`, SQLite):** resumable, up to 3 retries per job, dedup by base arXiv id, and a
  newer paper version replaces the old version's points.
* **PDF parsing (`ingestion/pdf_parser.py`):** PyMuPDF text spans, heading detection by font size / bold / known section
  names, producing `Segment(page, section, char offsets)`. Fallback OCR with Tesseract (`pytesseract`) for PDFs whose text
  layer is empty. Tesseract is installed in the Docker image.
* **Chunking (`ingestion/chunking.py`), semantic and section-aware:**
  1. Drop paragraphs shorter than **80 chars** and the entire **References** section.
  2. Within a run of the same section, greedily group consecutive paragraphs while the cosine similarity of the next
     paragraph to the group's running centroid is at least **0.45** (embeddings from `all-MiniLM-L6-v2`).
  3. Never merge across a section boundary, so a chunk's `section` is unambiguous.
  4. Split each group with LangChain `RecursiveCharacterTextSplitter` at **chunk_size = 500 chars, chunk_overlap = 70
     chars**, separators `["\n\n", "\n", ". ", " "]`.
  5. Discard fragments under **80 chars**.
* **Measured chunk length (characters):** mean 309.4, median 330, min 80, max 500, 10th percentile 114, 90th percentile 475.
* **Chunk identity and provenance:** `chunk_id = {arxiv_id}_{version}_{idx}`, plus `page`, `section`, `char_start`,
  `char_end`, `title`, `author`, `category`, `published_date`. Qdrant point ids are deterministic UUIDs derived from the
  `chunk_id`, so re-ingesting upserts instead of duplicating.
* **Embedding batch size for upsert:** 128.

---

## 4. Vector database specification

| Setting | Value |
|---|---|
| Engine | Qdrant (`qdrant-client` 1.19.1); embedded local mode, Docker, or Qdrant Cloud, chosen by env vars |
| Collection | `papers` (override with `QDRANT_COLLECTION`) |
| Vector dimension | 384 |
| Distance metric | Cosine |
| Index | HNSW, `m = 16`, `ef_construct = 100` |
| Points stored | 19,116 (one per chunk) |
| Raw vector footprint | about 29 MB (19,116 × 384 × 4 bytes; estimate, excludes payload and index) |
| On-disk size (embedded local store) | about 87 MB, including payloads |
| Payload indexes | `arxiv_id` (keyword), `category` (keyword), `published_date` (datetime), `author` (keyword), `section` (keyword) |
| Filtering | Filters (paper, category, date range, author, section) are pushed to Qdrant server-side |
| Quantization | None (explicit non-goal at this scale) |
| Mode switch | `QDRANT_URL` set → remote; unset → embedded at `data/qdrant_local`; `QDRANT_FORCE_LOCAL=1` forces local even if a URL is set |

Note: embedded local Qdrant ignores payload indexes (filters run in process); they only take effect against Docker or
Cloud. `scripts/sync_to_cloud.py` copies the local store to Qdrant Cloud without re-embedding.

---

## 5. RAG architecture

**Type:** hybrid retrieval-augmented generation with a post-generation verifier. Dense and sparse retrieval are fused with
rank-based fusion, reranked with a cross-encoder, and the generated answer is screened sentence by sentence by a layered
citation guard (§6).

### 5.1 Query path

```
browser ─ POST /query/stream (SSE) ─▶
  rate limit (per IP) + length cap
   ─▶ input guard (regex + jailbreak classifier) ── rejected: no retrieval, no LLM call
   ─▶ [optional LLM query rewrite] ─▶ hybrid retrieve: dense (Qdrant) + BM25 ─▶ RRF fusion
   ─▶ cross-encoder rerank ─▶ screen retrieved text for injection (drop injected chunks)
   ─▶ generate (Groq chain → Ollama), retry once on error / empty output
   ─▶ citation guard (per sentence: valid [N] → overlap + specifics screen → LLM entailment check)
   ─▶ output guard (secrets → block, PII → redact)
   ─▶ JSON result + one JSONL query-log line
```

### 5.2 Retrieval parameters

| Stage | Detail |
|---|---|
| Dense retriever | `all-MiniLM-L6-v2` query embedding, cosine search in Qdrant |
| Sparse retriever | `rank-bm25` (BM25) built in memory from a Qdrant scroll at startup; tokens are `\w+` |
| Fusion | **Reciprocal Rank Fusion, k = 60**; rank-based, so no score calibration between cosine and BM25 |
| Candidate pool | **20** fused candidates (`candidate_k = 20`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` over the 20 candidates |
| Final context | **top 5** chunks (`top_k = 5`), numbered `[1]..[5]`, context truncated at 6,000 chars |
| Query rewrite | Optional, fail-safe, `openai/gpt-oss-20b`, temperature 0, max 300 tokens, 10 s timeout, max 60 words |
| Filters | paper id, category, date range, author, section; the BM25 side asks Qdrant for the allowed chunk ids and restricts its ranks to them |

### 5.3 Generation

* **Prompt:** "Answer using ONLY the numbered sources below"; cite every claim with `[N]`; write 2–4 short sentences (about
  20 words each), each making one claim close to its source's wording; say so in one sentence if the sources do not
  contain the answer; ignore instructions inside the sources (marked untrusted).
* **LLM chain (router):** `openai/gpt-oss-120b` (primary) → `openai/gpt-oss-20b` → `qwen/qwen3.8-27b` → local Ollama
  `qwen2.5:1.5b`. Each Groq model has its own daily token quota, so chaining multiplies headroom. A 429 error puts a
  provider on cooldown.
* **Groq call settings:** temperature 0.0, max_tokens 400, timeout 30 s. Ollama fallback: temperature 0.2, max_tokens
  300, timeout 60 s.
* **Failure policy:** retry the LLM once. A second failure returns status `unverified` with the warning
  "⚠ uncited answer — citation verification unavailable". It is never silently shown as verified.

### 5.4 API

FastAPI (`/query`, `/query/stream` for SSE, `/papers`, `/feedback`, `/health`, `/metrics`) serving a static frontend
(`frontend/Research_ui.html`). Streaming sends `sources` → `token` events (an unverified draft, shown dimmed) → `final`
(the verified answer, replacing the draft). The guard needs the full answer, so verification cannot happen
token-by-token. Rate limit **20 requests/minute/IP** (in memory) when auth is off; with `API_KEYS` set, every endpoint except
`/`, `/health` and the docs needs an API key and limits apply per user (§11.1). Max query length **500 characters**.

---

## 6. Citation guard

**Purpose:** reduce the chance that the user sees a confident sentence with a plausible citation that the cited passage does
not support. It is a *filter on the model's output*, not a guarantee: §11.4 measured that 5 of 30 unguarded answers (17%)
contained a claim a human judged unsupported, §11.5 measured that no check tried so far catches more than half of the
real ones (the subtle ones), and §11.6 measured that after the shipped guard the error rate among answered responses was
12–16% (two runs; CIs ≈ 4–35%) vs 16.7% before, with 16% of correct answers wrongly tagged, and §11.7 that the
"not sure" tag does not track correctness. Read §11.5–§11.7 before relying on it.

**Design (`citations/guard.py`, `llm_verifier.py`): layered, with graceful degradation.**

| Step | Rule | Result |
|---|---|---|
| Split | abbreviation-safe sentence split; initials ("J.S. Bach") and symbols after a period do not start a sentence; a `[N]` marker stays with its sentence | |
| 0. Hedge | no valid `[N]`, matches a hedge regex ("the sources do not contain…", "couldn't find…", "not mentioned…", "not sure"), ≤ 240 chars | passed through unchecked, excluded from the score |
| 1. Citation | no valid `[N]` (N between 1 and the number of sources) and not a hedge | dropped |
| A1. Overlap floor | < **0.3** of the sentence's content words (length > 2) appear in the cited chunk(s) | dropped (`low_overlap`): catches wrong-chunk citations |
| A2. Specifics gate | any number, acronym, mixed-case or capitalised name of the sentence is missing from the cited chunks taken together (`GUARD_SPECIFICS_MIN` = 1.0) | dropped (`missing_specifics`, the missing items are returned) |
| B. Entailment | only for sentences that passed A. `GUARD_ENTAILMENT=llm` (default): one batched LLM call per answer (≤ 12 claims per call, 1,500-token output budget) asks, per claim, whether the cited passage(s) state or directly imply it, with every number, name, qualifier and comparison direction. **There is no fallback verifier** (a local NLI model was part of the design and was removed on request). `off` runs layer A only | not entailed → dropped (`not_entailed`) |
| Degraded | the LLM verifier could not run (provider down or rate-limited, unusable reply) or `off` | layer-A survivors are kept, `verification: lexical_only`, and a visible warning: "⚠ Entailment check unavailable — citations were screened only for matching words, numbers and names…" |

* **Score and abstain:** `faithfulness_score` = kept ÷ checkable sentences (hedges are on neither side); more than 50% of
  checkable sentences dropped, or none kept → the answer is *not* removed: status `low_confidence`, the model's original
  answer with the tag "⚠ Not sure — this answer could be wrong" and its sources. Otherwise status `ok` with the dropped
  sentences removed. A hedge-only answer is returned as `ok` with `hedged: true` and skips the "you must cite" retry.
* **Result fields:** `verification` (`llm` / `lexical_only`), `hedged`, per-sentence `details` (decision, reason, scores).
* **False refusals are tracked:** a hedge-only answer to an answerable question is a failure to answer, not a success.
  `hedged_rate` is in `/metrics`, and the evaluation reports `false_refusal_rate_on_answerable` and counts hedged answers
  as not answered (§10.4). Not yet measured on the full golden set.
* **Statuses returned overall:** `ok`, `low_confidence`, `unverified` (LLM failed twice), `abstained` (nothing retrieved),
  `rejected`, `blocked`.

**What it cannot do (measured, §11.5).** It does not prove entailment. The LLM verifier caught 2 of the 4 real human-labelled
unsupported sentences and missed the other 2: an inference the passage does not state, and a conflation of two different
uses of the same metric. Layer A stops cheap errors (wrong numbers, foreign names, wrong-chunk citations); layer B stops
contradictions and added claims; subtle over-reach still gets through. The verifier is the same model family as the
generator, so its mistakes can be correlated with the generator's, and with no fallback there is **no independent check**.
When the LLM is unavailable (including free-tier quota exhaustion, the binding capacity limit) the answer gets layer A
only, which caught 0 of 4 real unsupported sentences and 60% of the synthetic ones, and carries the `lexical_only` warning. The
verifier waits out a per-minute rate limit of up to 15 s once, but not a daily one; at the 8,000 tokens/minute free-tier cap
this degradation is common under load (§11.6).
Caveat: only the primary Groq model was evaluated as a verifier; if the router falls through to a smaller model (or local
Ollama), verification quality on that model is unmeasured.

**History.** Guard v1 used a local NLI model alone (threshold 0.5, 3-sentence premise windows, at most 3 windows per pair,
~6.1 s median per query): it dropped 22% of supported sentences and caught 2 of 4 unsupported ones (§11.4). A short-lived
v2 replaced it with embedding cosine plus a word-overlap check; §11.5 shows cosine carried no signal (AUC 0.45) and the
design was replaced by the layered guard above, and the NLI model was later removed from the codebase entirely (its
frozen scores remain in the evaluation data as evidence). The first NLI model tried, `cross-encoder/nli-deberta-v3-base`, scored almost
every correct multi-fact sentence near 0.00; a FEVER/ANLI-trained model, windowed premises and a "short single-claim
sentences" prompt fixed that.

---

## 7. Guardrails

| Check | Where | Model / method | Behaviour |
|---|---|---|---|
| Prompt injection / jailbreak in the query | Before retrieval | Regex pre-filter + `protectai/deberta-v3-base-prompt-injection-v2`, threshold **≥ 0.85** | Reject; no retrieval, no LLM call |
| Injection inside retrieved paper text | After rerank | Regex always; classifier on prose chunks only at **≥ 0.999** | Chunk quarantined |
| Secrets in output | After citation guard | Provider key patterns, private keys, JWTs, this process's own secret env values | Answer **blocked** |
| PII in output | After citation guard | Regex: email, phone, SSN, Luhn-valid card number | **Redacted** with a notice |

Why 0.999 on paper text: at 0.85 the classifier flagged about **4%** of benign paper chunks (numeric tables, reference
lists). The plan called for `guardrails-ai` Hub validators, which are not installable as plain PyPI packages, so the same
three checks are implemented locally with a Hugging Face classifier (no extra hosted dependency or API key).

---

## 8. Models used

| Role | Model | Notes |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384-d, used for chunking, indexing and queries |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 20 candidates → top 5 |
| Entailment verifier (citation guard) | the generator chain (Groq `gpt-oss-120b` → … ) | one batched call per answer, 1,500-token output budget |
| Jailbreak / injection classifier | `protectai/deberta-v3-base-prompt-injection-v2` | thresholds 0.85 (queries), 0.999 (papers) |
| Generator (primary) | Groq `openai/gpt-oss-120b` | temperature 0, max_tokens 400 |
| Generator fallback 1 | Groq `openai/gpt-oss-20b` | also used for query rewriting |
| Generator fallback 2 | Groq `qwen/qwen3.8-27b` | |
| Generator fallback 3 (local) | Ollama `qwen2.5:1.5b` | not available on a hosted CPU container |
| Evaluation judge | Groq `openai/gpt-oss-120b` via DeepEval | same family as the generator (see §12) |

The three local Hugging Face models (embedder, reranker, injection classifier) are downloaded at Docker build time (`scripts/download_models.py`) and run offline
(`HF_HUB_OFFLINE=1`) at runtime.

---

## 9. Evaluation

### 9.1 Golden set (`eval/data/golden_set.json`)
* **100 questions:** 80 answerable and 20 unanswerable (questions the corpus should not be able to answer).
* **Coverage:** gold passages from **57 distinct papers**.
* **Construction:** drafted automatically from corpus chunks (`eval/build_golden_set.py`); the source chunk is the gold
  chunk and a reference answer is stored. All 100 cases have `verification: auto` (machine-verified by an NLI check (since removed from the code)  plus a
  cross-checking LLM) and `reviewed: false` (no human review yet).
* **Regression subset:** 20 answerable cases flagged `regression` for the CI gate (60 other answerable cases are not).
* Each case stores: question, answerable flag, gold chunk ids, gold arXiv ids, reference answer, source page and section.

### 9.2 Evaluation components

> **Update (2026-10-14):** the DeepEval metrics described here were replaced by RAGAS (§11.8). The list below is the original design.

* **Retrieval ablation** at k = 5 over the 80 answerable cases (dense, BM25, hybrid RRF, hybrid + rerank).
* **Unanswerable test:** 20 cases, scored by whether the system refused or tagged the answer.
* **DeepEval generation metrics** with a Groq LLM judge, threshold 0.5, run on guarded and unguarded configurations:
  faithfulness, answer relevancy, contextual precision, contextual recall, contextual relevancy, plus a custom
  **citation correctness** metric.
* **Composite score** = mean of faithfulness, answer relevancy and citation correctness.
* **CI regression gate** (`tests/eval_gate/test_rag_quality.py`, GitHub Action `eval-gate`): per-case thresholds
  faithfulness ≥ 0.7, relevancy ≥ 0.6, citation ≥ 0.6, composite ≥ 0.7 (tunable via env vars).
* **Hand-label CLI** (`eval/hand_label.py`) to spot-check the judge against human verdicts.
* Long runs checkpoint per case and resume (`data/eval_runs/*_ckpt.jsonl`).

### 9.3 Test suite
* **443** offline tests pass plus **12** slow classifier tests deselected; live end-to-end smoke test **12/12** (UI, filters, SSE streaming,
  jailbreak rejected before any LLM call, rate limit 20/min/IP).
* Test packages mirror the code: ingestion, retrieval, citations, safety, storage, llm, pipeline, api, eval,
  security (prompt-injection suite) and the DeepEval gate.
* CI: `tests` workflow (unit tests on every PR and push to main, no keys needed) and `eval-gate` workflow (DeepEval on
  PRs touching backend/eval).

---

## 10. Results

> **Scope note.** Everything in §10 was measured on 2026-10-08 with citation guard v1 (NLI only). The guard was rebuilt on
> 2026-10-14 (§6, §11.5) and these end-to-end numbers (DeepEval scores, answer/abstention rates, latency) have **not** been
> re-run with it. Retrieval numbers (§10.1) do not depend on the guard.

### 10.1 Retrieval ablation (80 answerable cases, k = 5)

| Configuration | Chunk recall@5 | Chunk MRR | Paper recall@5 |
|---|---|---|---|
| Dense only | 0.750 | 0.627 | 0.963 |
| BM25 only | 0.900 | 0.830 | 0.988 |
| Hybrid (RRF) | 0.887 | 0.722 | 0.988 |
| **Hybrid + rerank** | **0.912** | **0.856** | **1.000** |

* RRF alone *lowers* MRR relative to BM25, because the weaker dense signal dilutes the top ranks. The cross-encoder
  repairs that, which is what makes the hybrid worth it.
* Caveat: golden questions were generated from the gold chunk's own wording and name specific entities, which favours
  BM25. Vaguer real questions should favour dense and hybrid more.
* Chunk-level recall is a lower bound (other chunks may also answer the question); paper-level recall is reported too.

### 10.2 Unanswerable questions
**19 of 20 (95%) correctly refused** (measured before the change that tags instead of refusing). The one miss (`u020`) was
answered with sentences that were each supported by cited passages but did not address the question's specific
condition: a relevance gap, not a fabrication. The system now returns a `low_confidence`-tagged answer for such
questions instead of a refusal; the eval counts a tag as handled.

### 10.3 Generation quality (DeepEval, partial; **superseded**)

> DeepEval's faithfulness scored every answer 1.0 including the human-labelled unfaithful ones (§11.8), so the faithfulness figures below carry no evidence of faithfulness. Kept as history.

The judge's free-tier daily token cap ran out after **9 of the planned 20** answerable cases (the run was paused with
exit code 3 and can resume). The guarded pipeline answered 7 of those 9 (**77.8%**); the other 2 abstained.

**On the 7 answered cases:**

| Metric | Guarded (full) | Unguarded baseline |
|---|---|---|
| Faithfulness | 1.000 | 1.000 |
| Answer relevancy | 0.952 | 1.000 |
| Citation correctness | 0.900 | 1.000 |
| Contextual precision | 0.957 | not measured |
| Contextual recall | 1.000 | not measured |
| Contextual relevancy | 0.318 | not measured |
| **Composite** | **0.951** | |

**On all 9 cases (abstentions counted as failures):**

| Metric | Guarded (full) | Unguarded baseline |
|---|---|---|
| Faithfulness | 0.778 | 1.000 |
| Answer relevancy | 0.852 | 0.889 |
| Citation correctness | 0.700 | 0.778 |
| Contextual precision | 0.828 | not measured |
| Contextual recall | 0.778 | not measured |
| Contextual relevancy | 0.293 | not measured |

* Contextual relevancy is low by construction: 500-character chunks contain mostly sentences irrelevant to any single
  question.
* At n = 7 on easy in-corpus questions, the guard shows **no measurable faithfulness gain, only a recall cost**. Its value
  is in adversarial and ungrounded cases this sample does not stress. All generation numbers are indicative, not
  statistically meaningful.
* The other 71 answerable cases have not been scored by the generation judge yet.

### 10.4 Behaviour summary
| Measure | Value |
|---|---|
| Answerable questions answered (guarded) | 7 / 9 = 77.8% |
| Unanswerable questions abstained | 19 / 20 = 95% |
| Run date | 2026-10-08 (guard v1) |
| False-refusal rate on answerable questions (hedge-only answers) | 1 / 30 = 3.3% (0.6–16.7%) on the 30 labelled questions with the shipped guard (§11.6); **not measured on the full 80-question golden set** |

### 10.5 Latency (CPU laptop, uncontended, 6 queries)
Median ≈ **8.6 s**, worst ≈ **15 s**. Stage medians: input guard 0.24 s, retrieve 0.14 s, rerank 0.5 s, injection
screen 1.2 s, generation 0.6 s, **NLI verification 6.1 s** (the bottleneck of guard v1; the layered guard's end-to-end latency has not been re-measured). Expect about 2× slower on a 2-vCPU
container. Cost: $0.

### 10.6 Lessons recorded during development
1. The first NLI model rejected nearly everything; fixed by a FEVER/ANLI model plus windowed premises.
2. The injection classifier false-positived on 4% of benign paper chunks at 0.85; paper text now uses 0.999.
3. Free-tier Groq limits were the real capacity constraint; long jobs now checkpoint and resume.
4. Eval bug: scoring abstentions as unfaithful made the guard look worse than no guard; abstentions are now excluded
   from quality metrics and reported as a recall cost.
5. An early smoke test wrote to the production vector DB; fixed so an explicit local path can never resolve to a remote.

---

## 11. Production-readiness work (added 2026-10-13)

Four items were added after the first evaluation: authentication with per-user limits, a managed-Qdrant snapshot with
a tested restore, hand labels to validate the judge and the NLI threshold, and failure-injection tests. All four are
implemented and tested.
Offline test suite after this work: **443 passed** (309 before 2026-10-13; 134 net new), 12 slow classifier tests deselected.

### 11.1 Authentication and per-user limits

| Item | Value |
|---|---|
| Mechanism | API key sent as `X-API-Key: <key>` or `Authorization: Bearer <key>` |
| Configuration | `API_KEYS="name:key[:per_minute[:per_day]],..."`; keys must be at least 16 characters; empty = auth OFF (local dev) |
| Defaults | `USER_RATE_LIMIT_PER_MIN=20`, `USER_DAILY_QUOTA=200` (queries per UTC day), overridable per user |
| Key comparison | constant-time (`hmac.compare_digest`) against every configured key |
| Brute-force throttle | bad-key attempts limited to `AUTH_FAIL_PER_MIN=10` per IP per minute, then HTTP 429 |
| Public paths | `/`, `/health`, `/docs`, `/redoc`, `/openapi.json` (the UI page itself carries no data) |
| Protected | `/query`, `/query/stream`, `/feedback`, `/papers`, `/metrics`, `/me` |
| Per-minute window | sliding 60 s window per **user** (not per IP), HTTP 429 + `Retry-After` |
| Daily quota | counted on `/query`, `/query/stream` only (feedback does not spend it); resets 00:00 UTC; the 429 message says when |
| `/me` | returns the caller's name, limits and `remaining_today` |
| CORS | CORS middleware is outermost, so 401/429 responses still carry CORS headers; preflight `OPTIONS` is never blocked |
| UI | the page prompts for the key on a 401 and keeps it in the browser's `localStorage` |
| Tests | 21 (`tests/test_api/test_auth.py`): parsing and bad specs, header and bearer, public paths, preflight, throttling, per-user isolation, daily quota and rollover, `/me` |

Limits of this design: keys are static strings from the environment (no signup, rotation or revocation without a
restart), and limiter state is in memory (resets on restart, not shared between workers).

### 11.2 Failure-injection tests

16 tests in `tests/test_failures/test_failure_injection.py`. They use real components where cheap: the real Groq
wrapper with a client that raises, a real Ollama client pointed at a closed port, a real `QdrantClient` pointed at a
dead port and at a TCP server that accepts and never replies, and the real `parse_pdf` on genuinely broken files.

| Failure | What was injected | Observed behaviour (asserted) |
|---|---|---|
| LLM down | Groq raises `ConnectionError`; Ollama not running | one clear error naming both providers; pipeline returns status `unverified` with the warning "⚠ uncited answer — citation verification unavailable", the top retrieved passages as the body and the sources listed; exactly one retry; HTTP 200, never a 500 |
| LLM down, streaming | same, over SSE | exactly one `retry` event, then a `final` event with status `unverified` |
| LLM partly down | primary fails, fallback healthy | fallback serves a normal verified answer (`ok`) |
| Qdrant down | connection refused | pipeline raises `RetrievalUnavailable` (no answer is invented, the LLM is never called); API returns **503**; `/health` returns 200 with `status: degraded`; the stream emits an `error` event |
| Qdrant hung | server accepts, never answers, client timeout 1 s | raises within 15 s instead of blocking forever |
| Qdrant recovers | outage then recovery on the same app object | 503 while down, a normal `ok` answer afterwards with no restart |
| Malformed PDF | garbage bytes, empty file, HTML saved as `.pdf`, truncated, header-only, AES-encrypted, blank page | `parse_pdf` raises a normal exception (`FileDataError`, `EmptyFileError`, `OcrUnavailable`, `ValueError`) or yields no text; the worker ends every bad job `failed` after 3 retries with an error message |
| Mixed batch | 7 bad PDFs plus 1 good | the good paper is embedded (1 embedded, 7 failed); nothing from a bad file reaches the index |
| Recovery | a failed file is replaced by a valid one | `reset_failed` then `run` embeds it |

Gaps this exposed and fixed: with Qdrant down the API used to return an unhandled 500 and `/health` crashed; both now
degrade as above (`RetrievalUnavailable`, `/health` "degraded", 503s). Not covered: the reranker failing,
and starting the service while Qdrant is down (the startup BM25 scroll would fail).

### 11.3 Managed Qdrant: snapshot and tested restore

Managed cluster: Qdrant Cloud, eu-central-1 on AWS; live collection `papers`. A free Qdrant Cloud cluster has no
managed backups, so the safeguard is an off-cluster snapshot plus a restore that has actually been run
(`scripts/qdrant_backup.py`, logic in `backend/app/storage/backup.py`).

| Command | What it does |
|---|---|
| `snapshot` | creates a server-side snapshot, downloads it, writes a `.json` manifest (sha256, size, point count, time), prunes old server-side snapshots (keeps 3) |
| `restore <file> --target <name>` | verifies the sha256, then uploads into a NEW collection; refuses the live collection and any non-empty target unless forced |
| `verify --target <name>` | compares a restored collection with the live one |
| `drill` | snapshot, restore into scratch collection `papers_restore_drill`, verify, delete scratch; never writes to `papers` |

**Restore drill result (run 2026-10-13 against the live cluster; `docs/restore_drill_latest.json`): PASSED.**

| Measure | Value |
|---|---|
| Snapshot size | 99,602,432 bytes (about 95 MiB) |
| Snapshot time (create + download) | about 17–22 s |
| Restore time (upload into scratch collection) | about 29–42 s (two drills) |
| Point count, source vs restored | 19,116 vs 19,116 |
| Vector config | 384-d, cosine, identical |
| HNSW config | m=16, ef_construct=100, identical |
| Payload indexes on restored collection | all 5 present (`arxiv_id`, `author`, `category`, `published_date`, `section`) |
| Random 200 points compared (vector and payload) | 0 missing, 0 vector mismatches, 0 payload mismatches |
| 20 real queries, top-5 on source vs restored | 20 / 20 identical |
| Live collection after the drill | still 19,116 points; scratch collection deleted |

Disaster recovery from the downloaded file: create a new cluster, set `QDRANT_URL` / `QDRANT_API_KEY`, then
`python scripts/qdrant_backup.py restore backups/<file> --target papers`. Unit tests (8, `tests/test_storage/test_backup.py`)
show the verifier catches a missing point, a tampered payload, a flipped vector and a snapshot failing its checksum, and
that restore refuses the live collection. Not done: a scheduled snapshot job (it is a manual command today) and
encryption of the downloaded file (`backups/` is git-ignored and Docker-ignored).

### 11.4 Hand-labelled validation of the judge and the NLI threshold

**Labels: the primary result uses human labels** (all 30 answers, labelled by the project owner in a chat session on 2026-10-13,
one case at a time, from the cached model answers with no new LLM calls and no scores shown). A second, independent
labelling by Claude (an LLM, blind to the scores) is kept for comparison; it is *not* a human label and is reported
separately below. To reproduce or extend: `python -m eval.label_sentences --labeler <name>` then
`python -m eval.label_validation report --labels eval/data/labels_human.jsonl eval/data/labels_claude.jsonl`.

**Design.** 30 answerable golden cases: the 9 already scored in the earlier DeepEval run plus 21 drawn at random (seed 13).
For each, the *unguarded* model answer was kept (so wrong sentences are present to be found), the NLI entailment score was
recorded for every cited sentence, and the DeepEval Faithfulness score (judge: Groq `gpt-oss-20b`) was recorded per answer.
Two label types: **sentence** (is a cited sentence supported by at least one passage it cites; the guard's own rule) and
**answer** (is every claim in the whole answer supported by the retrieved passages).

| Item | Count |
|---|---|
| Answers | 30: **25 labelled faithful, 5 not** (a023, a037, a047, a048, a053) |
| Sentences in those answers | 49; 42 carry a valid `[N]` citation (the rest are uncited); 2 answers have no cited sentence |
| Cited sentences analysed | 40 (**36 supported, 4 unsupported**); 2 further items excluded as sentence-splitter fragments (see below) |
| Unsupported sentences (human) | a023#1 (NLI 0.58), a047#1 (0.94), a048#2 (0.07), a053#2 (0.04); a fifth, the second sentence of a037, was split into fragments and is excluded from the statistics |

**Judge (DeepEval Faithfulness, pass = score ≥ 0.5) vs the human answer labels**

| Measure | Value |
|---|---|
| Judge score on all 30 answers | **1.0 for every answer** (mean 1.000) |
| Agreement with the human labels | 83.3% (25/30), 95% CI 66–93%; identical to the share of faithful answers, i.e. no better than "always pass" |
| Cohen's kappa | **0.0** |
| Unfaithful answers the judge passed | **5 of 5** |
| Faithful answers the judge flagged | 0 of 25 |
| Spearman vs share of supported sentences | undefined (the judge never varied) |

Reading: on this sample the judge has no discriminating power. The earlier headline "faithfulness 1.0" in §10.3 should be
read with that in mind: a judge that scores every answer 1.0 cannot confirm that answers are faithful. The five missed
answers are subtle (an unsupported inference, an added qualifier, a conflation of two different uses of a metric), which
is exactly what an LLM judge from the same family as the generator tends to miss.

**NLI citation guard v1 threshold (sentence-level; positive = "supported"), human labels** (guard v1 only: the runtime guard changed on 2026-10-14, see §11.5)

| Threshold | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 34 | 3 | 2 | 1 | 0.875 | 0.919 | 0.944 | 0.932 |
| 0.10 | 34 | 2 | 2 | 2 | 0.900 | 0.944 | 0.944 | 0.944 |
| 0.20 | 31 | 2 | 5 | 2 | 0.825 | 0.939 | 0.861 | 0.899 |
| 0.30 | 30 | 2 | 6 | 2 | 0.800 | 0.938 | 0.833 | 0.882 |
| 0.40 | 30 | 2 | 6 | 2 | 0.800 | 0.938 | 0.833 | 0.882 |
| **0.50 (current)** | **28** | **2** | **8** | **2** | **0.750** | **0.933** | **0.778** | **0.848** |
| 0.60 | 27 | 1 | 9 | 3 | 0.750 | 0.964 | 0.750 | 0.844 |
| 0.70 | 23 | 1 | 13 | 3 | 0.650 | 0.958 | 0.639 | 0.767 |
| 0.80 | 22 | 1 | 14 | 3 | 0.625 | 0.957 | 0.611 | 0.746 |
| 0.90 | 18 | 1 | 18 | 3 | 0.525 | 0.947 | 0.500 | 0.655 |
| 0.95 | 17 | 0 | 19 | 4 | 0.525 | 1.000 | 0.472 | 0.642 |

* At the guard's 0.5 threshold the guard **drops 8 of 36 supported sentences (22%)**; the false drops score 0.001–0.48
  (for example a correct, plainly-worded sentence scored 0.001). This agrees with the over-abstention described in §13.
* It **kept 2 of the 4 unsupported sentences** (NLI 0.58 and 0.94) and **caught 2** (0.07 and 0.04). The ones it kept were an
  inference not stated in the passage and a conflation of two different uses of Spearman's coefficient: NLI entailment does
  not detect claims that reuse the passage's words but extend or combine them.
* AUC = 0.785 (supported sentences usually outscore unsupported ones). Accuracy at 0.5 is 75.0% (95% CI 60–86%).
* Lowering the threshold raises recall (0.10 gives recall 0.944, accuracy 0.90, F1 0.944) but the unsupported sentences that
  score high stay in at every threshold below 0.6. With only 4 unsupported sentences, "best threshold" is not
  statistically meaningful, so **the threshold was not changed**. The data does support the direction of the known trade-off:
  0.5 over-drops correct sentences and does not buy much precision over lower values.
* What would settle it: a larger set that deliberately includes unsupported sentences (for example answers generated from
  deliberately wrong or mismatched passages).

**Human vs Claude labels (how far an LLM can stand in for a person here)**

| Level | Items compared | Agreement | Cohen's kappa |
|---|---|---|---|
| Sentence | 40 | 87.5% (35/40) | 0.22 |
| Answer | 30 | 80.0% (24/30) | 0.14 |

The LLM labelling was clearly more lenient than the human one: it passed 4 answers the human marked unfaithful (a023, a037,
a047, a048) and was stricter on 2 (a003, a007). Agreement is only slightly better than chance for the rare "not ok" class, so
LLM labels should not replace human labels for this task. The Claude-labelled results (judge passed 3/3 of its unfaithful
answers; NLI false-drop rate 24%, AUC 0.53) are in `eval/data/validation_report.json`; the human-labelled ones are in
`eval/data/validation_report_human.json`.

**A bug found while labelling.** The guard's sentence splitter cut text at abbreviations: "J.S. Bach" and "Data augm. + CL"
each produced a meaningless fragment that the guard would then have dropped. Fixed (initials and a following lowercase or
symbol no longer start a new sentence; "Dr.", "Prof." and similar added); 3 regression tests. The 2 fragments were excluded
from the statistics above (the human verdict on the full sentences is stored with them: a037 second sentence not supported,
a039 supported), so the numbers describe the splitter as it was when the set was built.

Files: `eval/label_validation.py` (build, show, report), `eval/label_sentences.py` (terminal labelling CLI),
`eval/data/label_set.jsonl` (blind items), `label_scores.jsonl` (hidden scores), `labels_human.jsonl`, `labels_claude.jsonl`,
`validation_report_human.json`, `validation_report.json`; tests in `tests/test_eval/test_label_validation.py` (7).

---

### 11.5 Citation guard rebuilt and stress-tested (2026-10-14)

**Starting point and critique.** A first replacement of the NLI guard (embedding cosine ≥ 0.35 plus a word-overlap check) was
faster but did not check entailment: in practice it checked word reuse, so it passes a sentence that reuses the passage's
words while extending or conflating its claim, which is how the 5 human-labelled unfaithful answers failed. Its thresholds
had been tuned on the same 40 labelled sentences that were used to judge it (4 unsupported ones, no held-out set), so its
77.5% vs 75.0% accuracy "win" over NLI meant nothing. It was withdrawn; nothing in this section presents any guard as proven.

**Method: a stress set that is not the tuning data** (`python -m eval.guard_stress build|score|report`).
Extra unsupported sentences were made by controlled *mutation* of the 36 human-labelled **supported** sentences, so the
label "unsupported" holds by construction, not by a model's say-so (201 items in all):

| Mutation | What it does | Items |
|---|---|---|
| number_swap | one number replaced by a different one | 11 |
| antonym_flip | a key word replaced by its opposite ("increases" → "decreases") | 6 |
| negation | "It is not the case that …" prepended (every word still comes from the passage) | 36 |
| qualifier | an unsupported extra claim appended ("…, and this holds in every setting that was tested") | 36 |
| conflation | a clause from a different paper appended, still citing the original chunk | 36 |
| wrong_chunk | the sentence checked against a chunk from a different paper (a wrong citation) | 36 |
| real_supported / real_unsupported | the human-labelled sentences | 36 / 4 |

Thresholds were chosen on the **real** supported sentences only (the highest threshold keeping false drops ≤ 20%) and
applied unchanged to the synthetic items. Caveat: synthetic errors are cleaner than real model errors; treat the catch rates
as a stress test, not a field rate. LLM-based rows were run on a sample (all 40 real items plus 11/6/12/12/12/12 synthetic
items per type, 105 in all) to stay inside the free-tier token budget.

**Candidate checks** (FD = real supported sentences wrongly dropped, of 36; "real" = real unsupported caught, of 4; the
mutation columns are the share of synthetic unsupported sentences caught):

| Check | FD | real | number | antonym | negation | qualifier | conflation | wrong chunk | all synthetic |
|---|---|---|---|---|---|---|---|---|---|
| Embedding cosine ≥ 0.45 | 19% | 0/4 | 27% | 0% | 19% | 19% | 25% | 100% | 38% |
| Word overlap ≥ 0.45 | 19% | 2/4 | 9% | 17% | 28% | 42% | 89% | 100% | 59% |
| Cosine AND overlap (the withdrawn v2) | 31% | 2/4 | 27% | 17% | 39% | 47% | 89% | 100% | 64% |
| Specifics (all numbers / names found) | 6% | 0/4 | 100% | 17% | 6% | 19% | 100% | 61% | 49% |
| NLI ≥ 0.4 (guard v1) | 19% | 2/4 | 91% | 67% | 83% | 97% | 78% | 100% | 89% |
| LLM, one claim per call | 8% | 2/4 | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

* **Cosine had no signal** on the real labels (AUC 0.45; NLI 0.785; overlap 0.73). Its only detections were wrong-chunk
  citations, which the overlap check also catches 100% of, so it added a parameter and nothing else. **Deleted.**
* **Word overlap alone is weak exactly where it matters:** number swaps 9%, negations 28%, qualifiers 42%.
* **The specifics check is cheap and precise** (6% false drops; 100% of number swaps and foreign-claim conflations).
* **Neither entailment-style check catches the real subtle errors.** On the 4 real unsupported sentences, NLI (since removed)
  and the LLM catch the *same* two (a048#2, a053#2) and both miss a023#1 (an inference the passage does not state) and a047#1
  (a conflation of two uses of one metric). No combination tried caught more than 2 of 4.

**What was shipped:** the layered guard of §6 (layer A = overlap floor 0.3 + specifics gate; layer B = batched LLM
verifier, no fallback). The shipped logic was evaluated end to end (`python -m eval.guard_stress report-final`; layer A
is the guard's own `screen_claim`, layer B uses cached model verdicts):

| Variant | FD | real | number | antonym | negation | qualifier | conflation | wrong chunk | all synthetic |
|---|---|---|---|---|---|---|---|---|---|
| Layer A only | 6% | 0/4 | 100% | 17% | 6% | 28% | 100% | 100% | 60% |
| **Layer A + LLM, batched (default)** | **17%** | **2/4** | 100% | 100% | 100% | 100% | 100% | 100% | **100%** (sample) |
| LLM batched alone | 11% | 2/4 | 100% | 100% | 100% | 100% | 100% | 92% | 98% (sample) |

* **Reading:** layering improves on word-reuse checks for the synthetic error types. Measured before the NLI model was
  removed: layer A + NLI had 25% false drops, 2 of 4 real, 95% synthetic, and NLI alone (guard v1) 19% / 2 of 4 / 89%, so the
  default (layer A + LLM, 17% false drops, 6 of 36) is not worse than what it replaced. It does **not** improve on the real
  subtle errors: 2 of 4, the same as NLI. The false drops of the default are 2 from layer A (a correct sentence using a derived word form, "U-shaped"
  vs "U-shape", and a full name that appears in a chunk it did not cite) and 4 from the LLM.
* **What the LLM-only choice costs:** its 100% is on the sample of mechanical mutations, which are easy; its real-error
  performance equals NLI's; it is the generator's own model family, so there is no independent check; and it costs an extra
  API call per answer (estimated +1–1.5k tokens, roughly a 30–40% smaller free-tier daily capacity; not measured end to
  end). Removing the NLI fallback also removes the only offline, zero-token verifier (370 MB of weights are gone from the
  image); when the LLM is down, answers fall to layer A with a warning (see §6).
* **A bug the evaluation found:** the verifier's replies were truncated (the shared 400-token output limit also covers the
  model's hidden reasoning), so 2 of 21 batched calls (about 10%) returned unusable JSON and would have fallen back every
  time. The verifier now requests 1,500 output tokens; afterwards 0 of 21 calls were unusable, and batched verdicts agree with
  the one-claim-per-call verdicts on 101 of 105 items.
* **Still true:** thresholds for layer A (0.3, all anchors) came from the 40 real labelled sentences, so they are
  in-sample for the real-sentence columns; the synthetic columns were not tuned on. There is no held-out set of real model
  errors, and 4 real unsupported sentences cannot support any claim about real-world catch rates. 17% (5 of 30; 95% CI
  roughly 7–34%) of unguarded answers containing an unsupported claim remains the number that matters, and no check here
  removes more than about half of the real ones.

**Also added with this change:** false-refusal tracking (hedge-only answers are counted as not answered; `hedged_rate` in
`/metrics`; `false_refusal_rate_on_answerable` in the evaluation), an `LLMRouter`/`GroqLLM`/`OllamaLLM.generate(max_tokens=…)`
override, a visible `lexical_only` warning when no entailment check can run, and tests (guard, verifier parsing, fallbacks,
stress mutations, false-refusal accounting).

---

### 11.6 What the user sees after the guard: error, answer and false-refusal rates (2026-10-14)

Measured by replaying the 30 human-labelled raw answers through the **shipped** guard (layer A + the LLM verifier on
`openai/gpt-oss-120b`, no fallback) and classifying each response exactly as `ResearchPipeline._assemble` does
(`python -m eval.post_guard_report`; per-case audit trails in `eval/data/post_guard_cases_run1.jsonl` and `post_guard_cases.jsonl`,
summary in `post_guard_report.json`). Sentence-level human labels decide what is an error. All intervals are Wilson 95%.
**The whole replay was run twice, with identical inputs and temperature 0, and the two runs differ on one case** (a048: the
verifier removed the unsupported sentence in run 2 and kept it in run 1), so the verifier is not deterministic and single-run
numbers carry that extra uncertainty.

| Outcome (n = 30 answerable questions) | Run 1 | Run 2 |
|---|---|---|
| **Pre-guard error rate** (raw answer has a human-labelled unsupported sentence) | 5 / 30 = 16.7% (7.3 – 33.6%) | same |
| **Answer rate** (clean answer, no "not sure" tag, not hedge-only) | 25 / 30 = 83.3% (66.4 – 92.7%) | same |
| **False-refusal rate** (answerable question, hedge-only answer) | 1 / 30 = 3.3% (0.6 – 16.7%) | same |
| Low-confidence rate (answer shown with a "not sure" tag) | 4 / 30 = 13.3% (5.3 – 29.7%) | same |
| **Post-guard error rate** (answered responses still containing an unsupported sentence) | 4 / 25 = 16.0% (6.4 – 34.7%) | 3 / 25 = 12.0% (4.2 – 30.0%) |
| Same, as a share of all 30 questions | 4 / 30 = 13.3% (5.3 – 29.7%) | 3 / 30 = 10.0% (3.5 – 25.6%) |
| Erroneous answers fixed or flagged by the guard | 1 / 5 = 20.0% (3.6 – 62.4%) | 2 / 5 = 40.0% (11.8 – 76.9%) |
| Correct answers wrongly tagged low-confidence | 4 / 25 = 16.0% (6.4 – 34.7%) | same |
| Supported sentences the guard wrongly removed | 6 / 37 = 16.2% (7.7 – 31.1%) | same |

**Reading it honestly**

* **The guard did not measurably reduce the error rate.** 16.7% before; 16.0% (run 1) or 12.0% (run 2) after, among answered
  responses, with heavily overlapping intervals. It fixed one or two of the five erroneous answers depending on the run. With
  n = 30 this neither shows an improvement nor proves there is none; it is not evidence that the guard works on real errors.
* **It has a visible cost.** 4 of the 25 correct answers (16%) came back tagged "not sure" (two because layer A rejected a
  correct sentence: "U-shaped" vs "U-shape", and a full name that was not in the cited chunk; one answer had no citations at
  all, which the live pipeline would retry once but a replay cannot; one rejected by the verifier), and 6 correct sentences
  (16%) were removed from answers that were otherwise kept.
* **The verifier's real-error catch rate is unstable.** On the isolated real unsupported sentences it caught 2 of 4; inside
  full answers, with its claims batched together, it caught 1 of 5 (run 1) or 2 of 5 (run 2). Four or five real unsupported
  sentences can only say "somewhere between none and about half".
* **The one false refusal is a retrieval miss, not a verifier decision:** a004 asks about a dataset that the retrieved passages
  do not mention, the model said so, and a human agreed the hedge was faithful. It counts as a false refusal because the golden
  set marks the question answerable. The rate is measured only on these 30 questions; the other 50 answerable golden cases have
  not been run through the guarded pipeline.
* **Not measured here:** the 80-question guarded run, latency, token cost, any model other than the primary verifier, and
  more than two replays (so the run-to-run spread is itself an estimate from two runs).

**A production finding from running it:** the free tier limits `gpt-oss-120b` to 8,000 tokens per minute, and the first replay
(about 4 calls a minute) hit 429s so often that 7 of 30 cases silently degraded to `lexical_only`. Those runs were discarded
and redone with pacing. Consequently the verifier now waits out a per-minute limit of up to 15 s once before giving up
(daily limits are not waited for), and under real traffic a share of answers will still carry the `lexical_only` warning.

**Definitions and limits.** Error = a shown sentence a human labelled unsupported. An *answered* response shows only the
guard's kept sentences; a *low-confidence* response shows the original answer, so unsupported sentences in it are visible but
flagged (none of the four low-confidence answers actually contained one). Thresholds for layer A were set on the same labelled
sentences, so layer A's behaviour here is in-sample. Three sentences of the low-confidence answer a007 were unlabelled at
sentence level (the human gave an answer-level "ok").

---

### 11.7 Answer-level faithfulness, label calibration, relevance and completeness (2026-10-14)

Computed on the same 30 cases for the text the user sees after the guard (run 2 of §11.6;
`python -m eval.answer_quality`, outputs `answer_quality_report.json` and `answer_quality_cases.jsonl`).

**1. Answer-level faithfulness (human labels)**

| Measure | Value | 95% CI |
|---|---|---|
| Raw answer faithful (no unsupported claim) | 25 / 30 = 83.3% | 66.4 – 92.7% |
| Post-guard response free of unsupported claims | 27 / 30 = 90.0% (run 1: 26 / 30 = 86.7%) | 74.4 – 96.5% |
| DeepEval faithfulness judge, for contrast | mean **1.000**, min 1.000 (n = 30) | uninformative: it scored every answer, including the 5 unfaithful ones, 1.0 |

**2. Is there a calibrated label?** No. The system attaches a label (`ok` / `low_confidence` / hedge) and a guard score
(kept ÷ checkable sentences). Neither tracks correctness in this sample:

| Label | Responses with no unsupported claim |
|---|---|
| answered (no tag) | 22 / 25 = 88.0% (70.0 – 95.8%) |
| low_confidence ("not sure") | **4 / 4 = 100%** (51 – 100%) |
| hedge-only | 1 / 1 |

* The three responses that still contained an unsupported claim were all untagged; none was flagged (0 / 3; 0 – 56%).
  Every response tagged "not sure" was in fact clean. The tag is currently noise.
* The guard score predicts correctness no better than chance: AUROC **0.494** (bootstrap 95% CI 0.278 – 0.778); mean score
  0.78 for clean responses vs 0.83 for responses with an error.
* A calibrated label (say High / Medium / Low with measured reliability per level) cannot honestly be fit on 3 to 5 erroneous
  responses. It needs a larger labelled sample (a few hundred answers) first.

**3. Relevance and completeness (LLM judge `gpt-oss-20b`; not human-labelled)**

| Score | Real post-guard responses (n = 30) | Positive control (reference answer as the response) | Negative control (response to a different question) |
|---|---|---|---|
| **Completeness** (share of the reference answer's key facts the response states) | **0.892** (bootstrap 95% CI 0.767 – 0.992) | 1.000 | 0.000 |
| **Relevance** (share of response sentences that directly help answer the question) | **0.933** (0.833 – 1.000) | 1.000 | 0.000 |

* By label: answered (n = 25) completeness 0.95, relevance 1.00; low_confidence (n = 4) 0.75 / 0.75; hedge-only (n = 1) 0 / 0
  (a hedge sentence is not counted as relevant content by this rubric, and it covers none of the reference facts).
* Lowest answered cases: a043 completeness 0.75; **a047 completeness 0.0 with relevance 1.0**, and a047 also contains one of
  the unsupported sentences, so a relevant-looking answer can be both incomplete and unfaithful.
* **What the controls do and do not show:** the judge is not trivially broken (it gives full credit to the reference and none
  to a wrong answer), but those controls are easy. They do not show that it can detect partial incompleteness or subtle
  irrelevance. The same provider family produced the answers and the faithfulness judge that scored every answer 1.0, the
  reference answers are machine-written (golden set, not human-reviewed), and no human labelled relevance or completeness.
  Treat these two scores as indicative until a human labels a subset (the existing terminal tool can be adapted).

---

### 11.8 DeepEval dropped, RAGAS adopted (2026-10-14)

**The problem.** The DeepEval faithfulness judge scored all 30 labelled answers 1.0, including the 5 that a human marked
unfaithful (§11.4, §11.7), so "faithfulness 1.0" in earlier results proved nothing.

**Diagnosis** (`eval/data/deepeval_diagnosis.json`; same judge, `gpt-oss-20b`, on the 5 human-unfaithful and 5 human-faithful answers):

| Answer (human label) | Claim verdicts | Default score | With `penalize_ambiguous_claims=True` |
|---|---|---|---|
| a023 (unfaithful) | yes, yes | 1.0 | 1.0 |
| a037 (unfaithful) | yes, yes | 1.0 | 1.0 |
| a047 (unfaithful) | **no**, yes, yes | 0.667 | 0.667 |
| a048 (unfaithful) | yes, **borderline**, yes | 1.0 | **0.667** |
| a053 (unfaithful) | yes, yes, yes | 1.0 | 1.0 |
| 5 faithful answers | all yes | 1.0 | 1.0 |

* **Cause 1, a real configuration bug:** by default DeepEval counts `borderline` ("not stated") claims as faithful
  (`penalize_ambiguous_claims=False`), and our code never set it. Fixing that moves flagged unfaithful answers from 1 of 5 to 2 of 5.
* **Cause 2, not fixable by a setting:** on three of the five unfaithful answers the judge marked *every claim "yes"*; and
  it is not even stable (a047 scored 1.0 in the first run and 0.667 here, at temperature 0). That is judge leniency and noise.
* **Decision:** by the stated rule (fix it, or drop DeepEval for RAGAS), the configuration fix is not enough, so DeepEval was
  removed from the evaluation code, the CI gate (`tests/eval_gate`), the workflow and `requirements-dev.txt`.
  (`eval/judge.py`, `deepeval_metrics`, `citation_metric` are gone; the old DeepEval numbers remain in this report as history.)

**RAGAS (`eval/ragas_metrics.py`, 0.4.3), judge `openai/gpt-oss-120b` at its default reasoning effort, via Groq's OpenAI-compatible API.**

| Requested | RAGAS metric used | Note |
|---|---|---|
| faithfulness | `Faithfulness` | share of the response's statements the retrieved contexts support |
| hallucination | 1 − `Faithfulness` (+ "any unsupported statement" flag) | RAGAS ships no separate hallucination metric; this is the statement-level complement |
| correctness | F1 of `FactualCorrectness` precision and reference coverage | precision = share of the response's claims the reference supports; coverage as below |
| completeness | reference-claim coverage (from `FactualCorrectness`'s reference-vs-response pass) | share of the reference answer's claims the response states (RAGAS's own recall is kept but not used: see the artefact note) |

Reference answers are machine-written (not human-reviewed), so correctness and completeness are relative to them.

**Does RAGAS fix the problem? Partly.** On the 30 raw answers, scored by the same human labels (`python -m eval.ragas_eval`):

| | Value |
|---|---|
| Human-unfaithful answers scoring below 1.0 | **2 / 5** (a023 = 0.75, a053 = 0.90); a037, a047, a048 scored 1.0 |
| Human-faithful answers scoring below 1.0 | **3 / 25** (a018 = **0.0**, a003 = 0.83, a007 = 0.86) |
| Mean faithfulness, unfaithful vs faithful answers | 0.930 vs 0.948 |
| AUROC (does a faithful answer outscore an unfaithful one?) | **0.632** (bootstrap 95% CI 0.42 – 0.87, includes 0.5) |
| DeepEval on 10 of the same cases, for contrast | flagged 1 / 5 (default) or 2 / 5 (penalising ambiguous); 0 / 5 false flags |

RAGAS removes the "always 1.0" symptom (scores vary and some real errors are flagged), but it does **not** validate as a faithfulness
detector: the confidence interval on AUROC includes chance, three of five real errors score 1.0, and it falsely flags 12% of faithful answers
(a018, a correct one-sentence answer, scored 0.0). The limit is the same as the guard's verifier: an LLM judge from the
generator's own model family approves subtle over-reach. Switching metric libraries cannot fix that; only a stronger or independent
judge, or human labels, can.

**Findings from making it work**
* **RAGAS 0.4.3 does not import here** (`langchain_community.chat_models.vertexai` was removed in langchain-community 0.4.x and RAGAS
  imports it at start-up; the dependency is unpinned, so any fresh install breaks). `load_ragas()` installs a stub for that one module,
  limited to the evaluation code, instead of altering installed packages. A regression test imports RAGAS for real (and fails, not skips,
  if the import breaks: a skip would have hidden it).
* **Reasoning effort matters:** at `low` effort the same judge scored the test answer 1.0 (missed); at the default effort it scored 0.71
  and 0.90 in two runs (caught). The default is used. Run as three separate RAGAS metrics this cost about 13k judge tokens per case
  (faithfulness 4.1k, precision 3.1k, recall 5.9k); RAGAS's recall mode already repeats the precision pass, so precision and recall are now
  computed from at most two claim passes (4 judge calls instead of 6, RAGAS's own formulas), about 10k per case (faithfulness 4.1k +
  factual correctness 5.9k); recomputing only the coverage pass for all 30 cases cost about 60k tokens. 30 cases is still roughly 300k tokens against a 200k/day, 8k/minute free-tier limit. The 10 cases scored before
  this change used the two-metric path; the definitions are identical.
  Runs are therefore cached per score and resumable; they stop cleanly (exit 3) when the daily quota runs out.
* **RAGAS faithfulness is not deterministic** either: the same answer scored 0.90 and 0.71 at temperature 0 on two runs.
* **Bugs in my own first adapter, found by running it** (all fixed, with regression tests): a fresh event loop per call broke the HTTP
  client ("Event loop is closed") from the second call; a score cache keyed on `id()` could return another case's scores; a stray backspace
  character from a scripted edit silently broke a new hedge regex pattern (the repository was scanned and nothing else was affected).
* **CI gate:** `tests/eval_gate/test_rag_quality.py` now scores faithfulness, correctness and completeness with RAGAS under
  `pytest -m eval`. Its thresholds (0.7 / 0.3 / 0.5, composite 0.5) are provisional floors, not validated targets.

**Live end-to-end check of the pipeline after the swap** (real models, real Qdrant, `TestClient`, 55 s to build, 19,116 chunks):
`/health` ok; an answerable question returned a correct cited answer (`status ok`, `verification llm`); a second answerable question
came back `low_confidence` (guard score 0.33); an off-corpus question was answered "none of the provided sources contain…" (see below);
a prompt-injection query was rejected with no LLM call; the injected chunk in the corpus was quarantined; the streaming path ended in a
`final` event; `/metrics` reported the traffic. Two issues surfaced and were handled:
* the model's honest "none of the provided sources contain information…" was not recognised as a hedge, so it was tagged `low_confidence`
  and not counted in `hedged_rate`; hedge patterns now cover "none of the … sources/passages/documents" and "the sources … do not …";
* under token-rate pressure (my parallel evaluation job) the router fell back to `gpt-oss-20b` for generation and for verification, so the
  verifier is not always the model that was evaluated.

**RAGAS results on what the user sees after the guard, all 30 cases, one judge (`gpt-oss-120b`)** (`python -m eval.ragas_eval run|report`;
scores in `ragas_eval_scores.jsonl`, summary in `ragas_eval_report.json`; means with bootstrap 95% intervals over cases)

| Metric | Mean | 95% CI | n |
|---|---|---|---|
| **Faithfulness** | **0.937** | 0.862 – 0.995 | 30 |
| **Hallucination** (1 − faithfulness) | **0.063** | 0.005 – 0.138 | 30 |
| Responses with at least one unsupported statement | 4 / 30 = 13.3% | 5.3 – 29.7% | 30 |
| **Correctness** (F1 of precision vs reference and reference coverage) | **0.532** | 0.372 – 0.708 | 30 |
| **Completeness** (share of the reference's claims the response states) | **0.783** | 0.650 – 0.900 | 30 |

By the guard's label: answered (n = 25) faithfulness 0.97, correctness 0.60, completeness 0.82; low-confidence (n = 4) 0.71 / 0.25 / 0.75
(all four were human-faithful answers); the one hedge-only answer 1.0 / 0 / 0 (it answers nothing, so completeness 0 is right).
21 of 30 responses cover every claim of the reference (completeness 1.0).

**A measurement artefact found and corrected (2026-10-15).** The first full pass reported correctness 0.55 and completeness 0.59 and gave
**12 of 30 cases an exact 0.0 for both**, including plainly correct answers such as a002, a014 and a033 that restate the reference. Cause:
* RAGAS's own recall is `tp/(tp+fn)` where `tp` counts *response* claims the reference supports. If none is supported, recall is 0 even when
  the response covers every claim of the reference. Here it happened because the references are terse extracts of a passage while good
  responses restate the question's context ("the drop in accuracy for target-class samples is most pronounced…"), which a strict
  claim-by-claim check against the short reference marks "unsupported".
* Fix: completeness is now *reference-claim coverage* (the share of the reference's claims that the response states, from the pass that
  was already being run) and correctness is the F1 of precision and that coverage. Mean completeness moves from 0.589 (RAGAS recall,
  12 zeros) to **0.783** (4 zeros: a004 the hedge-only answer, a007 the uncited answer, and a014 and a047). RAGAS's own recall is still
  stored as `fact_recall` for transparency, and a regression test pins the case where precision is 0 but coverage is 1.
* **What is still unreliable:** precision against the terse reference still has **12 exact zeros** (mean 0.52), so the correctness figure
  of 0.53 is an **underestimate**: it penalises correct answers for restating the question's context. a014 (a correct, human-faithful
  answer) still gets completeness 0, which is judge strictness. For "is the answer right", faithfulness (is it supported by what was
  retrieved: 0.94) is the more trustworthy number here; correctness against a human-written, question-aware reference would need
  a human-reviewed reference set.
* The first debug run also showed this judge is **not stable** on borderline claims: the same claim for a002 was judged supported in one
  run and unsupported in later ones.

**Other caveats.** Faithfulness on the guarded text does not agree strongly with the human labels: the 3 responses a human marked as still
containing an unsupported sentence were not clearly lower than the 27 clean ones (AUROC 0.61, CI 0.41 – 0.93). Reference answers are
machine-written. Completeness here (0.78) differs from the earlier own-judge completeness (0.89, §11.7): different rubric and judge.

**Bottom line.** RAGAS is installed, working, tested and wired into the evaluation loop, the CI gate and the docs, and it replaces DeepEval.
Faithfulness/hallucination and completeness are usable indicators with wide intervals; correctness is a lower bound; none is ground truth.
With this judge RAGAS catches 2 of the 5 human-unfaithful answers and falsely flags 3 of 25 faithful ones.

---

### 11.9 Repository cleanup and live end-to-end test (2026-10-15)

**Scope rule:** keep only what takes part in the live system, testing, evaluation, the citation guard or the guardrails; comment out
dead code (using the repository's `# [DEAD CODE]` marker) instead of deleting it; remove files that serve none of those purposes.
Before anything was deleted a snapshot of every source, doc and small data file (145 files, 0.4 MB) was taken, because nothing in the
working tree had been committed yet.

**How dead code was found:** an AST scan for definitions no production code references (tests excluded), an import graph of every
module (every module is either imported by production code or is a runnable entry point: none orphaned), and a scan for unused imports
and unreferenced configuration constants (none).

| Action | What | Why |
|---|---|---|
| Commented out | legacy `/ask` endpoint and `AskRequest` (and its two tests) | the UI uses `/query/stream`; only tests called it |
| Commented out | `OllamaLLM.clear_cache` / `cache_stats` (and their tests; a test fixture now clears the dict directly) | no production caller |
| Commented out | 4 unused imports (`math`, two `time`, `point_id`; a fifth was in the removed script) | dead |
| Removed | `scripts/relabel_sections.py` | one-off maintenance for a past labelling change; part of none of the five areas |
| Removed | empty `.deepeval/` folder | DeepEval was dropped (§11.8) |
| Removed | 3 of 4 Qdrant snapshots in `backups/` (380 MB to 95 MB) | identical copies of one data set; the newest, the one verified by the restore drill, is kept |
| Archived | the DeepEval-era generation scores (`data/eval_runs/generation_ckpt_deepeval_legacy.jsonl`, `eval/reports/legacy_deepeval_generation.json`) | otherwise the dashboard would label old DeepEval numbers as RAGAS; `latest.json` keeps only the retrieval ablation |
| Fixed | stale references to a `ResearchAgent` class that does not exist | documentation |
| Not touched | a 113 MB demo screen recording in the repo folder (git-ignored, not part of the repository) | a personal file: delete it by hand if unwanted |

**New: `tests/live/test_live_e2e.py` (opt-in: `python -m pytest -m live tests/live`).** Ten tests with the *real* models, the real Qdrant
cluster (read-only), the real Groq chain for both generation and the guard's verifier, the real guardrails, auth and rate limits, the
streaming endpoint, the metrics log and a RAGAS faithfulness pair. Query and feedback logs are redirected to a temp directory so the test
never pollutes runtime data. They are skipped in a normal test pass so nobody spends tokens by accident.

**Results (2026-10-15):** offline suite **440 passed, 0 failed** (10 skipped = the opt-in live tests; 3 fewer than before because three
tests of commented-out dead code were commented out with it); **live suite 10 / 10 passed in 112 s**: corpus and `/health` live
(more than 10,000 chunks, at least 90 papers); a cited answer with arXiv provenance links; an off-corpus question not answered
confidently; a prompt-injection query rejected with no retrieval or LLM call; the stream ending in a `final` event; metrics logging the traffic;
a credential-shaped output withheld and PII redacted; the real verifier rejecting a negated claim and layer A rejecting a wrong number;
auth returning 401 without a key and per-user limiting after exactly 3 requests; and RAGAS scoring a supported answer above one with an
invented claim. The earlier live run of this system found and fixed three problems (hedge phrases, an event-loop bug, a stray control
character), all covered by regression tests.

---

## 12. Deployment

* **Single container (`Dockerfile`):** `python:3.12-slim`, CPU-only `torch==2.8.0`, Tesseract OCR, non-root user,
  models baked into the image, FastAPI + frontend served by uvicorn on port **7860** (or `$PORT` if set), healthcheck on
  `/health`.
* **Run locally:**
  ```
  docker build -t research-agent .
  docker run -p 7860:7860 --env-file .env research-agent
  ```
  Secrets via env: `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY` (leave the Qdrant ones empty for embedded storage), and
  optionally `API_KEYS` to turn on authentication (§11.1) and `GUARD_ENTAILMENT` (`llm` default, `off`) for the citation guard (§6).
* **Resource estimates (not measured):** image about 4–5 GB, disk about 5 GB, RAM about 1.5–2 GB at runtime (torch plus
  three local Hugging Face models). It will not fit in 512 MB.
* **Observability:** JSONL query log with stage timings, tokens and cost; p50/p95 through `/metrics`.
* **Key libraries:** FastAPI 0.141.1, uvicorn 0.32.1, sentence-transformers 6.0.1, transformers 5.16.1, torch 2.8.0,
  qdrant-client 1.19.1, groq 0.37.1, rank-bm25 0.2.2, PyMuPDF 1.26.7, pytesseract 0.3.13, langchain-text-splitters
  1.1.2, arxiv 4.0.1.

---

## 13. Known limitations

* **The guard does not guarantee faithfulness (the main limitation).** 17% of unguarded answers in the human-labelled sample
  contained an unsupported claim, and no check tried catches more than 2 of the 4 real unsupported sentences: unstated
  inferences and conflations of two facts get through (§11.5). The checks it does make are cheap-error filters
  (numbers, names, wrong-chunk citations) plus an entailment check whose real-error performance equals a local NLI model's.
* **The LLM verifier shares the generator's model family and has no fallback**, so its errors can correlate with the
  generator's and nothing independent backs it up. When the LLM is unavailable (including quota exhaustion) citations get
  layer A only, with a warning. Verification costs an extra API call per answer (estimated, not measured), which shrinks the
  already binding free-tier capacity. Only the primary Groq model was evaluated as a verifier.
* **False refusals are possible and not yet measured:** if the model hedges when the sources do contain the answer, the
  user gets an honest-sounding non-answer with status `ok`. `hedged_rate` and `false_refusal_rate_on_answerable` now
  track it. Measured on 30 labelled questions: 1 of 30 (3.3%; 0.6–16.7%); the other 50 golden cases are unrun.
* **The verifier is not deterministic:** two identical replays (temperature 0) differ on one of 30 cases, so any single-run guard
  number carries extra uncertainty (§11.6).
* **The "not sure" label is not calibrated** (§11.7): every tagged answer in the sample was clean, none of the untagged errors
  was flagged, and the guard score is at chance for predicting correctness.
* **Relevance and completeness (§11.7) come from an unvalidated LLM judge** with easy controls only; **RAGAS faithfulness/hallucination
  are noisy** (AUROC 0.63 against human labels, 3 of 25 faithful answers falsely flagged, non-deterministic across identical runs) and
  RAGAS correctness is an underestimate (it penalises correct answers that restate the question's context against terse reference answers; 12 of 30 precision scores are exactly 0) (§11.8).
* **The guard has a measurable cost:** 4 of 25 correct answers (16%; 6–35%) came back tagged "not sure" and 6 of 37
  correct sentences (16%) were removed, while it fixed 1–2 of 5 real errors (§11.6).
* **Layer A is blunt:** the specifics gate dropped 2 of 36 correct sentences (a derived word form, and a name that
  appears in a chunk the sentence did not cite); the overlap floor is loose by design.
* **Thresholds are in-sample:** 0.3 / all-anchors were set from the 40 real labelled sentences; the synthetic
  stress set is mechanical and cleaner than real errors; there is no held-out set of real model errors.
* **Tagged answers can still be wrong.** A `low_confidence` answer is unverified model output; the tag and sources are
  the safeguard.
* **Grounded is not on-topic.** The guard checks support by a retrieved passage, not whether the question is in the
  corpus's subject area ("capital of France" was answered "Paris" from an illustrative sentence in a paper).
* **Prompt-injection defence is layered, not complete;** paraphrased attacks can pass the regex, and the paper-text
  classifier is deliberately conservative. A paper that legitimately discusses prompt injection will have those chunks
  quarantined.
* **Judge bias:** the LLM judge (DeepEval earlier, RAGAS now) is the same provider and family as the generator, so absolute scores are likely
  optimistic; relative comparisons are more trustworthy.
* **Evaluation scope:** generation metrics cover only 9 answerable cases; ground truth is machine-verified, not
  human-labelled; no case has been human-reviewed. The judge/NLI validation (§11.4) uses human labels but contains only 5 unfaithful answers and 4 unsupported
  sentences: it shows the judge never discriminated, but it cannot prove a better threshold.
* **Judge:** on the 30 labelled answers the faithfulness judge scored every answer 1.0 and passed all 5 human-labelled unfaithful ones, so
  the 1.0 faithfulness in §10.3 does not by itself demonstrate faithful answers.
* **Parsing:** section detection is heuristic; math and tables extract poorly; the References section is dropped;
  `Appendix` is a large catch-all (about 27% of chunks); OCR needs the Tesseract binary.
* **Retrieval:** BM25 is held in memory and built at startup (fine at about 19k chunks); single English-only 384-d embedder.
* **Capacity:** each Groq free-tier model has its own 200,000-token daily cap (about 100 queries/day/model, 1.5–2.5k
  tokens per answered query), so the chain gives roughly 3× headroom; after that queries fail until quotas roll over.
* **Operations:** rate limits and daily quotas are in memory (reset on restart, not shared across workers); API keys are static
  environment strings (no rotation or revocation without a restart); snapshots are a manual command, not scheduled; no
  quantization; no admin flow for adding corpora. Not failure-tested: the reranker failing, and starting the service while
  Qdrant is down (the entailment chain itself degrades visibly, §6).

Full detail: `docs/architecture.md`, `docs/known_limits.md`, `docs/case_study.md`.

---

## 14. Repository layout

```
backend/app/        api, auth, config, pipeline, services
  ingestion/        arxiv_fetcher, pdf_parser, chunking, job_queue, worker
  storage/          qdrant_store, backup (snapshot / restore / verify)
  retrieval/        embedder, hybrid_retriever, reranker, query_rewriter
  citations/        guard (layered screen + entailment chain), llm_verifier (batched LLM entailment),
                    formatter (numbered context + arXiv links)
  llm/              groq_llm, local_llm (Ollama), router (fallback chain)
  safety/           guardrails (injection screen, secret/PII output guard)
  telemetry/        observability (query log, p50/p95)
frontend/           Research_ui.html
eval/               runner, generation (DeepEval), judge, golden, metrics, hand_label, label_validation, label_sentences,
                    guard_stress + guard_stress_scoring (mutation stress test), post_guard_report (error / answer /
                    false-refusal rates), answer_quality (faithfulness, label calibration, relevance, completeness),
                    ragas_metrics + ragas_eval (RAGAS faithfulness / hallucination / correctness / completeness), dashboard,
                    build_golden_set, run_all
  data/             golden_set.json, label_set / label_scores / labels_human / labels_claude (.jsonl), validation_report*.json,
                    guard_stress_set / _scores / _llmbatch (.jsonl), guard_stress_report.json, guard_final_report.json
  reports/          latest.json + dashboard.html
scripts/            ingest_papers, ask, sync_to_cloud, qdrant_backup, download_models
tests/              unit tests mirroring the packages + security suite + failure injection (test_failures) + RAGAS CI gate (eval_gate)
                    + tests/live (opt-in end-to-end test with the real models, Qdrant and Groq)
docs/               architecture, known_limits, case_study, restore_drill_latest.json
backups/            Qdrant snapshots (git-ignored)
data/               runtime state (git-ignored): Qdrant store, PDFs, job queue, logs, eval checkpoints
```
