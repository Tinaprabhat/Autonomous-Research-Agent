# Architecture

Autonomous Research Agent v2: a citation-grounded Q&A system over 99 arXiv papers (19,116 chunks).
Numbers and measured results are in [`info.md`](info.md); the story of how the design got here is in [`case_study.md`](case_study.md).

**Scope.** One FastAPI process, one worker, one Qdrant collection, a free LLM tier. API-key auth with per-user limits.
Hosted on AWS ECS (US$100 free credit). The binding capacity limit is the Groq free tier (8,000 tokens/min and 200,000 tokens/day per model), not the code.

---

## 1. High-level design

Six planes: ingestion (offline), serving (online), guard and guardrails (inside serving), storage, operations, and evaluation (offline).

```
╔═══════════════════════════ A. INGESTION  (offline, resumable) ═════════════════════════════╗
║                                                                                            ║
║  arXiv API ─▶ harvest metadata ─▶ SQLite job queue ─▶ download PDF ─▶ parse (PyMuPDF;      ║
║  (8 topic       data/manifest.json   pending>downloaded>   data/raw_papers     OCR fallback ║
║   queries,                          parsed>chunked>embedded                    Tesseract)   ║
║   >=3 s apart)                      (failed after 3 retries)                       │       ║
║                                                                                    ▼       ║
║        Qdrant upsert (batch 128) ◀─ embed (MiniLM, 384-d) ◀─ chunk: semantic, section-aware ║
║        point id = UUID(chunk_id)     data/processed_chunks     500 chars, 70 overlap,       ║
║                                                                drop References, min 80 chars║
╚══════════════════════════════════════════════│═════════════════════════════════════════════╝
                                                ▼
              ┌──────────────────────────────────────────────────────────────────┐
              │ B. STORAGE   Qdrant Cloud  collection "papers"                   │
              │  19,116 points · 384-d cosine · HNSW m=16 ef=100 · 5 payload     │
              │  indexes (arxiv_id, category, author, section, published_date)   │
              │  snapshot + tested restore (backups/) · SQLite jobs.db · JSONL   │
              │  query/feedback logs                                             │
              └───────────────▲──────────────────────────────────────────────────┘
                              │ dense search + id lookups + scroll (BM25 build at start-up)
╔═════════════════════════════│═══ C. SERVING  (online, one request) ═════════════════════════╗
║                                                                                            ║
║  Browser UI (static page) ── POST /query/stream (SSE) or /query ──▶                        ║
║                                                                                            ║
║  [1] EDGE         API-key auth ─▶ per-user per-minute window + per-UTC-day quota           ║
║                   ─▶ query length cap (500) ─▶ per-IP limiter (when auth is off)           ║
║                                   │                                                        ║
║  [2] INPUT GUARDRAIL              ▼   regex + injection classifier (>= 0.85)               ║
║                   rejected ─▶ status "rejected": no retrieval, no LLM call                 ║
║                                   │                                                        ║
║  [3] RETRIEVAL                    ▼                                                        ║
║      optional query rewrite (gpt-oss-20b, fail-safe)                                       ║
║        ├─ dense: embed query ─▶ Qdrant cosine search (filters pushed server-side) ─ 30     ║
║        └─ sparse: BM25 over in-memory index (allowed ids from Qdrant)  ───────────── 30    ║
║                   ─▶ Reciprocal Rank Fusion (k=60) ─▶ 20 candidates                        ║
║                   ─▶ cross-encoder rerank (ms-marco-MiniLM-L-6-v2) ─▶ top 5                ║
║                   ─▶ context screen: regex on all, classifier (>= 0.999) on prose chunks;  ║
║                      injected chunks quarantined                                           ║
║                   nothing found ─▶ status "abstained"                                      ║
║                                   │                                                        ║
║  [4] GENERATION                   ▼   numbered context [1]..[5] (<= 6,000 chars)           ║
║      LLM router: gpt-oss-120b ▶ gpt-oss-20b ▶ qwen3.8-27b ▶ Ollama qwen2.5:1.5b            ║
║      temperature 0, max 400 tokens; 429 puts a provider on cool-down                       ║
║      retry once on error/empty ─▶ 2nd failure: status "unverified" + visible warning       ║
║                                   │                                                        ║
║  [5] CITATION GUARD               ▼   (section 2)                                          ║
║      hedge ▶ [N] check ▶ overlap + specifics (layer A) ▶ ONE batched LLM entailment (B)    ║
║      >50% dropped ─▶ "low_confidence" (answer kept, tagged "Not sure")  else ─▶ "ok"       ║
║      LLM check unavailable ─▶ layer A only, verification = "lexical_only" + warning        ║
║                                   │                                                        ║
║  [6] OUTPUT GUARDRAIL             ▼   secrets ─▶ block ("blocked") · PII ─▶ redact         ║
║                                   │                                                        ║
║  [7] RESPONSE                     ▼   answer, status, citations (arXiv links, page,        ║
║      section), faithfulness_score, hedged, verification, warning, timings, tokens          ║
║      SSE order: sources ─▶ token (unverified draft, dimmed) ─▶ final (verified, replaces)  ║
╚═════════════════════════════════════════│══════════════════════════════════════════════════╝
                                           ▼
      D. OPERATIONS: JSONL query log (stage timings, tokens, cost) ─▶ GET /metrics (p50/p95,
         status counts, hedged_rate) · GET /health (ok | loading | degraded) · Qdrant outage ─▶ 503
         · backup/restore script · API-key auth + limits · Docker image (models baked in)
         · deployed on AWS ECS (ap-southeast-2)

╔═══════════════════════════ E. EVALUATION & QUALITY  (offline) ═════════════════════════════╗
║  golden set (100 Qs: 80 answerable, 20 not) ─▶ retrieval ablation (recall@5, precision@5,  ║
║  MRR) · 30 human-labelled answers ─▶ guard replay + stress test (201 items) · RAGAS        ║
║  (faithfulness, hallucination, completeness, correctness; judge gpt-oss-120b) ·            ║
║  pytest: 440 offline + 10 live end-to-end · failure-injection (16) · CI: tests, eval-gate  ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Citation guard in detail

```
model answer (sentences with [N] markers)
   │  split: abbreviation-safe ("J.S. Bach", "et al." do not split; [N] stays with its sentence)
   ▼  for each sentence
 [0] Hedge?   no [N], <= 240 chars, matches "the sources do not contain..."      yes ─▶ pass, unscored
 [1] Cited?   needs a valid [N] (1..number of sources)                           no  ─▶ DROP no_citation
 [A1] Overlap floor   < 0.3 of content words found in the cited chunk(s)         ─▶ DROP low_overlap
 [A2] Specifics gate  every number / acronym / name must appear in the cited
      chunks (GUARD_SPECIFICS_MIN = 1.0)                                         ─▶ DROP missing_specifics
   ▼  survivors, all together
 [B]  ONE batched LLM call (<= 12 claims, 1,500-token budget): "does the cited passage state or
      directly imply this, with every number, name, qualifier and direction?"    no  ─▶ DROP not_entailed
   ▼
 score = kept / checkable;  > 50% dropped or none kept ─▶ low_confidence;  else ok
```

* **No fallback verifier.** A local NLI model was part of the design and was removed. If the LLM cannot run
  (provider down, rate-limited, unusable reply, or `GUARD_ENTAILMENT=off`), layer A's survivors are kept and the response
  is marked `verification: lexical_only` with a visible warning. A per-minute 429 is waited out once (up to 15 s); a daily one is not.
* **What it is.** A filter on mechanical errors (wrong numbers, names, citations, contradictions), not a proof of faithfulness.
  Measured limits: `info.md` section 10.4 and `case_study.md` sections 3.3 and 3.4.

## 3. Guardrails

| Check | Where | Method | Behaviour |
|---|---|---|---|
| Prompt injection in the query | before retrieval | regex pre-filter + `protectai/deberta-v3-base-prompt-injection-v2`, >= 0.85 | reject; no retrieval, no LLM call |
| Injection in retrieved paper text | after rerank | regex always; classifier on prose chunks only at >= 0.999 (at 0.85 about 4% of benign chunks were flagged) | chunk quarantined |
| Secrets in output | after the citation guard | provider key patterns, private keys, JWTs, this process's own secret env values | answer blocked |
| PII in output | after the citation guard | email, phone, SSN, Luhn-valid card number | redacted with a notice |
| Abuse and fair use | edge | API keys (constant-time compare), per-user minute window and UTC-day quota, bad-key throttle per IP, query length cap | 401 / 429 with `Retry-After` |

The written plan named `guardrails-ai` Hub validators. They are not installable as plain PyPI packages, so the same three checks
are implemented locally in `safety/guardrails.py` with a local Hugging Face classifier (no extra hosted dependency or key).

## 4. Status model

| Status | Meaning |
|---|---|
| `ok` | answer shown; unsupported or uncited sentences removed (`hedged: true` if it is an honest "the sources don't say") |
| `low_confidence` | more than half the sentences failed; original answer shown, tagged "Not sure", with sources |
| `unverified` | generation failed twice; top passages shown with a warning, never presented as verified |
| `abstained` | retrieval found nothing |
| `rejected` | prompt injection caught at the input |
| `blocked` | credential-shaped output withheld |

## 5. Components

| Concern | Module | Notes |
|---|---|---|
| Config | `backend/app/config.py` | env-driven; accepts the `Quadrant_*` spellings already in `.env` |
| API, middleware | `api.py`, `auth.py` | routes, CORS (outermost), auth, per-user and per-IP limiters, SSE |
| Orchestration | `pipeline.py`, `services.py` | one code path for the API, streaming and the evaluation harness |
| Harvest / queue | `ingestion/arxiv_fetcher.py`, `job_queue.py`, `worker.py` | SQLite `pending > downloaded > parsed > chunked > embedded / failed`, retries <= 3, dedup by base id, a newer version replaces old points |
| PDF parse | `ingestion/pdf_parser.py` | PyMuPDF spans, heading detection, `Segment(page, section, offsets)`; Tesseract OCR for empty text layers |
| Chunking | `ingestion/chunking.py` | semantic grouping (cosine to running centroid >= 0.45), never across sections; `chunk_id = {arxiv_id}_{version}_{idx}` |
| Store | `storage/qdrant_store.py` | embedded-local / Docker / Cloud by env; an explicit local path can never resolve to a remote |
| Backup | `storage/backup.py`, `scripts/qdrant_backup.py` | snapshot, restore into a new collection, verify, drill |
| Retrieval | `retrieval/embedder.py`, `hybrid_retriever.py`, `query_rewriter.py`, `reranker.py` | MiniLM embedder; dense + BM25 + RRF; optional rewrite; cross-encoder |
| LLM | `llm/groq_llm.py`, `local_llm.py`, `router.py` | model chain with per-model quotas; 429 classified per-minute vs daily; retry policy lives in the pipeline |
| Citation guard | `citations/guard.py`, `llm_verifier.py`, `formatter.py` | layered guard, batched verifier, citation and provenance formatting |
| Guardrails | `safety/guardrails.py` | input, context and output guards |
| Observability | `telemetry/observability.py` | JSONL log, p50/p95, hedged rate, false refusals |
| Frontend | `frontend/Research_ui.html` | one static page: filters, streaming answer, source cards, feedback, API-key prompt |
| Evaluation | `eval/` | golden set, retrieval ablation, RAGAS, guard stress test, post-guard report, label tools, dashboard |
| Tests | `tests/` | offline suite, failure injection, opt-in live end-to-end (`-m live`), CI eval gate |

## 6. Why these choices

* **Provenance before storage.** Citation links (`arxiv.org/abs/{id}`, `pdf/{id}v{n}#page=N`) are built from fields written at chunk time.
* **RRF instead of union-by-presence.** Rank-based fusion needs no calibration between cosine and BM25 scores. RRF alone lowers MRR against BM25;
  the cross-encoder repairs it, which is why the reranker is worth its latency.
* **Filters pushed to Qdrant.** Dense filters run server-side; the BM25 side asks Qdrant for the allowed ids and restricts its ranks to them.
* **Answer always, honestly labelled.** A rejected-by-guard answer is still shown, tagged "Not sure", with sources. Only an injection-rejected query,
  a credential-leaking answer or an empty retrieval gets no answer.
* **Fail visibly.** Every degraded path (LLM down, verifier unavailable, Qdrant down) produces a status or warning, never a silent "verified".
* **One verifier, no pretend fallback.** The NLI fallback caught the same real errors as the LLM, dropped 22% of good sentences, added about 6 s and 370 MB, and was removed.
  The cost is that nothing independent backs the LLM check (`case_study.md` section 3.3).
* **Checkpoint everything long-running** (ingestion, golden build, evaluation), because the free-tier quotas are the real limit.

## 7. Deployment shape

One container (`Dockerfile`): `python:3.12-slim`, CPU torch 2.8.0, Tesseract, non-root user, the three Hugging Face models baked in at build time and
run offline (`HF_HUB_OFFLINE=1`), uvicorn serving the API and the static page on port 7860 (or `$PORT`), a `/health` healthcheck with a 180 s start period.
Secrets arrive as environment variables: `GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`, and optionally `API_KEYS` (turns auth on), `GUARD_ENTAILMENT`.
Needs about 1.5 to 2 GB RAM and a start-up of 25 to 55 s (pipeline build) up to about 3 min (full app). Deployed on AWS ECS (ap-southeast-2) from this image,
paid from a US$100 free credit; the Hugging Face Space and Render deployments were tried and removed from the repository. Image size, ECS task size and
ECS latency are not yet measured. Single worker only, and a single ECS task: rate limits and quotas are in memory, and a redeploy means about three minutes of downtime.
