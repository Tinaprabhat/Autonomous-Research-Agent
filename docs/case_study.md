# Case study: a citation-grounded research assistant on a $0 stack

## Problem
Answer questions about ~100 arXiv papers so that every claim can be traced to a passage. The failure to design
against is not "no answer" but a *confident answer with a plausible citation that doesn't support it*.

## What was built (v1 → v2)
v1 was a FAISS + union-by-presence hybrid + a local LLM, with a "citation guard" that reused a relevance
cross-encoder (so a relevant-but-wrong citation passed) and silently fell back to the raw answer. v2:

| Area | v1 | v2 |
|---|---|---|
| Chunk identity | per-paper counter (collides across papers) | `{arxiv_id}_{version}_{idx}` + page/section/char offsets |
| Store | FAISS flat | Qdrant: HNSW `m=16, ef_construct=100`, 5 payload indexes, server-side filters |
| Fusion | union of dense ∪ BM25 | Reciprocal Rank Fusion (k=60) + cross-encoder rerank |
| Verification | relevance score; raw-answer fallback | per-sentence NLI entailment; uncited/unsupported sentences dropped; if >50% fail, the answer is still shown but tagged "not sure / could be wrong" |
| Failure policy | silent | retry LLM once, then a visible "⚠ uncited answer — citation verification unavailable" |
| Safety | none | injection screen on queries *and* retrieved paper text; secret blocking + PII redaction on output |
| Ingestion | one script | SQLite job queue (retries, resumable, version-aware dedup), OCR fallback |

## Results (all numbers from this repo's runs, 2026-10-08)

**Corpus:** 99 of 100 harvested papers ingested (1 genuine arXiv 404), 19,116 chunks, 57+ papers covered by the golden set.

**Retrieval ablation** — 80 answerable golden cases, k=5 (`eval/reports/latest.json`):

| Config | Chunk recall@5 | Chunk MRR | Paper recall@5 |
|---|---|---|---|
| dense | 0.750 | 0.627 | 0.963 |
| BM25 | 0.900 | 0.830 | 0.988 |
| hybrid (RRF) | 0.887 | 0.722 | 0.988 |
| **hybrid + rerank** | **0.912** | **0.856** | **1.000** |

The reranker is what makes hybrid worth it: RRF alone *lowers* MRR relative to BM25 (a weak dense signal dilutes
the top ranks), and the cross-encoder repairs that. Caveat: golden questions were generated from the gold chunk's
own wording and name specific entities, which favours BM25; real, vaguer questions should favour dense/hybrid more.

**Unanswerable questions** (measured *before* the answer-with-warning change, when the system refused) — 20 questions the
corpus should not answer: **19/20 (95%) correctly refused.** The system now returns a `low_confidence`-tagged answer instead
of a refusal for these; the eval counts a tag as handled. The one miss
(`u020`) was answered with sentences that are each supported by cited passages but don't address the question's
specific condition — a relevance gap, not a fabrication.

**Generation quality (DeepEval, Groq judge) — partial, n=9 answerable.** The judge's free-tier daily token cap was
exhausted after 9 of the planned 20 answerable cases (the run checkpoints and resumes). On the 7 cases the guarded
pipeline answered: faithfulness 1.0, answer relevancy 0.95, citation correctness 0.90, contextual precision 0.96,
contextual recall 1.0 (contextual relevancy 0.32 is low by construction: 500-char chunks contain mostly sentences
irrelevant to any single question). The guarded pipeline *answered* 7 of 9 (78%); the other 2 abstained.
The unguarded baseline scored faithfulness 1.0 on those same 7 — at n=7 on easy in-corpus questions the guard shows
**no measurable faithfulness gain, only a cost in recall**; its value is in the adversarial/ungrounded cases this
sample doesn't stress. Treat all of this as indicative, not statistically meaningful.

**Latency** (CPU laptop, uncontended, 6 queries): median ≈ 8.6 s, worst 15 s. Stage medians: input guard 0.24 s,
retrieve 0.14 s, rerank 0.5 s, injection screen 1.2 s, generation 0.6 s, **NLI verification 6.1 s** (the bottleneck).
Expect ~2× slower on a 2-vCPU Hugging Face Space. Cost: $0 (free tiers).

**Tests:** 331 offline unit tests + 12 classifier tests; live end-to-end smoke (12/12): UI, filters, SSE streaming,
jailbreak rejected before any LLM call, rate limit (20/min/IP).

## What went wrong along the way (and what it taught)
1. **The first NLI model rejected ~everything.** `cross-encoder/nli-deberta-v3-base` gave ≈0.00 entailment to
   correct multi-fact sentences; the system abstained on 3 of 4 answerable questions. A FEVER/ANLI-trained model,
   sentence-window premises and a "short single-claim sentences" prompt fixed it. Measure a verifier on real
   answers before trusting a threshold.
2. **The injection classifier false-positived on 4% of benign paper chunks** (numeric tables, citations) at the
   obvious 0.85 threshold, and it was caught only because retrieved chunks were quarantined in a live run. Query
   screening and paper-text screening need different operating points.
3. **Free-tier limits were the real capacity constraint.** Each Groq model has a 200k-token *daily* cap
   (~100 queries/day/model). The golden-set build died mid-run on it and *lost 60 verified cases held in memory*;
   everything long-running (ingestion, golden build, DeepEval) now checkpoints per item and resumes. The LLM router
   chains three Groq models plus Ollama and cools down rate-limited providers.
4. **Eval methodology bug:** scoring an abstention ("Not enough evidence…") as an unfaithful answer made the guard
   look *worse* than no guard. Abstentions are now excluded from quality metrics and reported as a recall cost.
5. **A test wrote to the production vector DB.** An early smoke test used a config where `.env` pointed at the
   cloud cluster. Fixed structurally (an explicit local path can never resolve to a remote; tests force local).

## Limits
See [`known_limits.md`](known_limits.md): machine-verified (not human) ground truth, same-family LLM judge, strict
NLI over-abstains on abstractive answers, injection defence is layered not complete, grounded ≠ on-topic,
free-tier quotas cap daily users.
