# Architecture

Autonomous Research Agent v2 — a citation-grounded Q&A system over ~100 arXiv papers.
Scope: 100 papers (~40k vectors), ~10 users, $0 infrastructure (Groq free tier, Qdrant Cloud free
cluster, Hugging Face CPU Space).

## Query path

```
 browser ── POST /query/stream (SSE) ───────────────────────────────────────────────┐
                                                                                     ▼
 ┌─────────────┐   ┌──────────────┐   ┌─────────────────────────────┐   ┌───────────────────┐
 │ rate limit  │──▶│ input guard  │──▶│ (rewrite) → hybrid retrieve │──▶│ rerank → screen   │
 │ per IP, len │   │ jailbreak    │   │ Qdrant dense + BM25 → RRF   │   │ cross-encoder;    │
 │ cap         │   │ regex+model  │   │ filters pushed to Qdrant    │   │ drop injected     │
 └─────────────┘   └──────┬───────┘   └─────────────────────────────┘   │ chunks            │
                          │ rejected: no retrieval, no LLM call          └─────────┬─────────┘
                                                                                   ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │ generate (Groq gpt-oss-120b → Ollama fallback); on error/empty output retry ONCE          │
 │   ├─ 2nd failure ─▶ status "unverified" + "⚠ uncited answer — citation verification        │
 │   │                 unavailable" (never silently presented as verified)                    │
 │   └─ success ─▶ NLI citation guard (per sentence: needs valid [N] AND entailment ≥ 0.5)    │
 │                   ├─ >50% of sentences fail ─▶ status "low_confidence": the model's answer  │
 │                   │     is still shown, tagged "⚠ Not sure — this answer could be wrong"     │
 │                   └─ else ─▶ status "ok": verified, cited; unsupported sentences removed     │
 └──────────────────────────────────────────────┬──────────────────────────────────────────┘
                                                ▼
                          output guard: secrets → block · PII → redact
                                                ▼
                          JSON result + one JSONL line in the query log
```

Streaming: `sources` event right after retrieval → `token` events (an **unverified draft**, shown
dimmed) → `final` event with the verified answer, which replaces the draft in the UI. The guard
needs the full answer, so verification cannot happen token-by-token.

## Components

| Concern | Module | Notes |
|---|---|---|
| Config | `backend/app/config.py` | env-driven; accepts the `Quadrant_*` spellings already in `.env` |
| PDF parse | `ingestion/pdf_parser.py` | PyMuPDF spans → heading detection → `Segment(page, section, offsets)`; OCR fallback (pytesseract) for empty text layers |
| Chunking | `ingestion/chunking.py` | semantic merge of paragraphs (running centroid), never across sections, 500-char cap; `chunk_id = {arxiv_id}_{version}_{idx}` |
| Job queue | `ingestion/job_queue.py`, `worker.py` | SQLite `pending→downloaded→parsed→chunked→embedded / failed`, retries ≤3, resumable, dedup by base id, version upgrade replaces old points |
| Store | `storage/qdrant_store.py` | HNSW `m=16, ef_construct=100`; payload indexes on `arxiv_id, category, published_date, author, section`; embedded-local / Docker / Cloud by env |
| Retrieval | `retrieval/hybrid_retriever.py` | dense (Qdrant, server-side filters) + BM25 → Reciprocal Rank Fusion (k=60); optional LLM query rewrite (fail-safe) |
| Rerank | `retrieval/reranker.py` | `ms-marco-MiniLM-L-6-v2` cross-encoder after RRF |
| Guardrails | `safety/guardrails.py` | see below |
| Generation | `llm/groq_llm.py`, `local_llm.py`, `router.py` | chain `gpt-oss-120b → gpt-oss-20b → qwen3.8-27b → Ollama` in the router (each Groq model has its own daily quota; 429s are classified as per-minute vs daily); call-level retry once in the pipeline |
| Verification | `citations/guard.py` | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, 3-sentence sliding windows over the cited chunk, max entailment |
| Orchestration | `pipeline.py`, `services.py` | one code path for API, streaming and `ResearchAgent` |
| API | `api.py` | `/query`, `/query/stream`, `/papers`, `/feedback`, `/health`, `/metrics`; serves the static UI |
| Observability | `telemetry/observability.py` | JSONL log: stage timings, tokens, cost (default $0), p50/p95 via `/metrics` |
| Eval | `eval/` | DeepEval + Groq judge, retrieval ablation, hand-label CLI, static dashboard |

## Why these choices

* **Provenance before storage.** Citations link to `arxiv.org/abs/{id}` (and `pdf/{id}v{n}#page=N`)
  built from fields written at chunk time; Qdrant payload filters need the same fields indexed.
* **RRF instead of union-by-presence.** Rank-based fusion needs no score calibration between cosine
  similarity and BM25 and rewards agreement between the two rankers.
* **Filters pushed to Qdrant.** The dense side filters server-side. The BM25 side asks Qdrant for the
  allowed chunk ids and restricts BM25 ranks to them — no post-hoc Python filtering of results.
* **Answer always, honestly labelled.** A verified answer comes back clean and cited. If the guard rejects most of an
  answer, the user still gets the model's answer, but tagged "not sure / could be wrong" (status `low_confidence`) with
  the sources it pointed at, so they can check. Only a query rejected by the input guard, a credential-leaking answer,
  or a retrieval that found no sources at all gets no answer. Within a verified answer: uncited sentences are dropped,
  unsupported cited sentences are dropped. Guard calibration notes are in
  `known_limits.md`.

## Guardrails (plan step 4.5)

| Check | Where | Behaviour |
|---|---|---|
| Jailbreak / prompt injection (query) | before retrieval | regex pre-filter + `protectai/deberta-v3-base-prompt-injection-v2` (≥0.85) → reject, no LLM call |
| Injection inside retrieved paper text | after rerank | regex always; classifier only on prose chunks at ≥0.999 (tables/citations false-positive at 0.85) → chunk quarantined |
| Secrets in output | after citation guard | provider key patterns, private keys, JWTs, and this process's own secret env values → answer **blocked** |
| PII in output | after citation guard | email / phone / SSN / Luhn-valid card → **redacted** with a notice |

**Deviation from the written plan:** the plan names `guardrails-ai` Hub validators
(`DetectJailbreak`, `DetectPII`, `SecretsPresent`). Those are not installable as plain PyPI packages
(they require a Hub token and `guardrails hub install`), so the same three checks are implemented locally in
`safety/guardrails.py` with the same fail-safe semantics, using a Hugging Face classifier run locally (no new
hosted dependency or API key). The `guardrails-ai` package itself is not a dependency.

## Deployment shape

One container (`Dockerfile`): uvicorn serves API + `frontend/Research_ui.html` on port 7860 (Hugging Face
Spaces, Docker SDK, CPU Basic). Models are pre-downloaded at build time. Secrets arrive as env vars:
`GROQ_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`. A scheduled GitHub Action hits `/health` (which runs a
Qdrant `count`) to keep the Space awake and register activity against Qdrant Cloud's idle policy.
