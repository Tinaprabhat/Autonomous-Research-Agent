# Autonomous Research Agent

Citation-grounded question answering over a curated set of ~100 arXiv papers.
Every sentence in an answer must carry a `[N]` citation that links to the arXiv paper (and PDF page),
and is screened against the cited passage (matching numbers, names and words, then an LLM entailment check) — unsupported sentences are removed. If most of an answer
fails the check you still get the answer, but tagged **"⚠ Not sure — this answer could be wrong"** with its sources
listed so you can check them yourself.

* Retrieval: Qdrant (dense) + BM25 → Reciprocal Rank Fusion → cross-encoder rerank, with server-side filters
  (category, author, section, date range).
* Generation: Groq `gpt-oss-120b` → `gpt-oss-20b` → `qwen3.8-27b` (each has its own free daily quota) → Ollama; streamed to the UI over SSE.
* Safety: jailbreak/prompt-injection screening (queries **and** retrieved paper text), secret blocking and PII
  redaction on output, per-IP rate limit and query-length cap.
* Evaluation: RAGAS (Groq judge; DeepEval was dropped, see docs/case_study.md section 3.2) + recall@k/MRR ablation + a terminal hand-labelling tool.

Details: [`docs/architecture.md`](docs/architecture.md) · limits: [`docs/case_study.md`](docs/case_study.md)
· write-up: [`docs/case_study.md`](docs/case_study.md)

## Quick start (local, no Docker needed)

```powershell
pip install -r requirements-dev.txt
copy .env.example .env      # then fill GROQ_API_KEY (leave QDRANT_URL empty for embedded local storage)
```

`QDRANT_URL` empty → embedded Qdrant in `data/qdrant_local` (one process at a time).
`QDRANT_URL` + `QDRANT_API_KEY` set → Qdrant Cloud or Docker (`docker run -p 6333:6333 qdrant/qdrant`).
Add `QDRANT_FORCE_LOCAL=1` (or `--local` on the scripts) to use local storage even when a URL is configured.

### 1. Ingest papers

```powershell
python scripts/ingest_papers.py --local harvest --max-papers 100   # queue arXiv papers (rate-limited, deduped)
python scripts/ingest_papers.py --local run                        # download → parse → chunk → embed (resumable)
python scripts/ingest_papers.py --local status                     # per-state counts + failures
python scripts/ingest_papers.py --local retry-failed
python scripts/ingest_papers.py verify                             # against a server: fails loudly if payload indexes are missing
```

Edit `DEFAULT_QUERIES` / `DEFAULT_CATEGORIES` in `scripts/ingest_papers.py` (or pass `--queries`, `--categories`) to
change the topic mix. `data/manifest.json` lists every paper and is the recovery path if the vector DB is lost.

### 2. Ask / serve

```powershell
python scripts/ask.py --local "How does MAML adapt to a new task?"
uvicorn backend.app.api:app --port 8000        # UI at http://localhost:8000
```

API: `POST /query`, `POST /query/stream` (SSE), `GET /papers`, `POST /feedback`, `GET /health`, `GET /metrics`.
Request body: `{"query": "...", "top_k": 5, "filters": {"category": "cs.LG", "date_from": "2023-01-01"}}`.
Result `status` is one of `ok` (verified, cited), `low_confidence` (answer shown with a "not sure / could be wrong" tag),
`unverified` (LLM failed twice — visible warning), `rejected` (input guard), `blocked` (credential-like output) or
`abstained` (retrieval found no sources at all).

### 3. Test

```powershell
python -m pytest -m "not slow"        # ~270 offline tests (fakes; never touches cloud Qdrant)
python -m pytest -m slow              # downloads/runs the prompt-injection classifier
```

### 4. Evaluate

```powershell
python -m eval.run_all --local --gen-limit 20 --allow-partial-golden   # unattended: golden set -> retrieval -> DeepEval -> dashboard
python -m eval.runner report --local          # re-aggregate from checkpoints (no LLM calls)
python -m eval.runner abstention --local      # unanswerable cases only (no judge tokens)
python -m eval.dashboard                      # bakes eval/reports/dashboard.html from eval/reports/latest.json
python -m eval.hand_label data/eval_runs/answers_latest.jsonl   # optional y/n human spot-check, resumable
python -m pytest -m eval tests/eval_gate/test_rag_quality.py     # CI regression gate (RAGAS; needs keys)
```

* `eval/data/golden_set.json` (100 cases: 80 answerable with gold chunk + reference answer, 20 unanswerable) is **machine-verified
  (an NLI check, since removed, + a second LLM), not human-reviewed**; timestamped backups live in `data/eval_runs/`.
* Everything long-running checkpoints per item (`data/eval_runs/*_ckpt.jsonl`, `golden_partial.jsonl`). If a Groq model's
  **daily token quota** is exhausted the stage exits with code 3 and **re-running the same command resumes** (quotas roll
  over after ~24 h). The latest results are in [`docs/case_study.md`](docs/case_study.md).

## Run with Docker (local)

```
docker build -t research-agent .
docker run -p 7860:7860 --env-file .env research-agent
```

Set `GROQ_API_KEY`, `QDRANT_URL` and `QDRANT_API_KEY` in `.env` (leave the Qdrant vars empty for embedded local storage).
To populate Qdrant Cloud once from your machine (no re-embedding): `python scripts/sync_to_cloud.py` (dry run), then
`python scripts/sync_to_cloud.py --yes`. The UI is served at http://localhost:7860.

**Auth (optional):** set `API_KEYS="name:key[:per_minute[:per_day]],..."` to require an `X-API-Key` header and enforce
per-user limits (see `.env.example`). **Backups:** `python scripts/qdrant_backup.py drill` snapshots the Qdrant collection,
restores it into a scratch collection, verifies it and cleans up; `snapshot` / `restore` / `verify` are available separately.
**Citation guard:** layered (number/name/overlap screen, then an LLM entailment check; no fallback verifier); set `GUARD_ENTAILMENT=llm|off`. It filters, it does not guarantee faithfulness: see `docs/info.md` §7 and `docs/case_study.md` §3.3, and `python -m eval.guard_stress` for the stress test.
**Judge validation:** `python -m eval.label_sentences --labeler <name>` to hand-label, then
`python -m eval.label_validation report --labels eval/data/labels_human.jsonl`. Full report: [`docs/info.md`](docs/info.md).

## Layout

```
backend/app/                 service code (one concern per package)
  api.py  config.py  pipeline.py  services.py     entrypoint, settings, query orchestration, wiring
  ingestion/    arxiv_fetcher  pdf_parser  chunking  job_queue  worker
  storage/      qdrant_store
  retrieval/    embedder  hybrid_retriever  reranker  query_rewriter
  citations/    guard + llm_verifier (citation checks)  formatter (numbered context + arXiv links)
  llm/          groq_llm  local_llm (Ollama)  router (model fallback chain)
  safety/       guardrails (injection screen, secret/PII output guard)
  telemetry/    observability (query log, p50/p95)
frontend/       Research_ui.html (served by the API)
eval/           evaluation code only
  data/         golden_set.json            reports/   latest.json + dashboard.html
  runner  generation  judge  golden  metrics  hand_label  dashboard  build_golden_set  run_all
scripts/        CLI utilities: ingest_papers  ask  sync_to_cloud  qdrant_backup  download_models
tests/          mirrors the packages above (test_ingestion, test_retrieval, test_citations, test_safety, ...)
                + test_security (prompt-injection suite) + eval_gate (RAGAS CI gate)
docs/           architecture  info  case_study
data/           runtime state, git-ignored (Qdrant store, PDFs, job queue, logs, eval checkpoints)
```
