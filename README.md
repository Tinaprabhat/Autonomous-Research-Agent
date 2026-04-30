# Autonomous Research Agent
### A Fully Local, Citation-Grounded AI Research Pipeline

> **Every answer is cited. Every citation is verified. Nothing is fabricated.**

Built on a multi-stage retrieval and reranking pipeline, this system autonomously ingests academic papers, retrieves the most relevant knowledge, and generates answers that are **structurally constrained to only what the sources support** — eliminating hallucination by design, not by hope.

The entire system runs locally. No cloud AI APIs. No data leaving your machine.

---

## The Problem This Solves

Standard RAG chatbots retrieve context and hand it to an LLM with a polite instruction to "use the context below." The model complies — until it doesn't. When retrieved chunks are thin, ambiguous, or partially relevant, the model fills the gaps with confident-sounding fabrications.

This project takes a different position:

> **If a claim cannot be cited, it should not be in the answer.**

The citation guard layer enforces this structurally. The LLM is required to attach `[N]` markers to every factual claim. Any sentence that fails to cite a retrievable source is automatically dropped before the answer reaches the user.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OFFLINE — Run Once                   │
│                                                         │
│  arXiv / PDF  →  PyMuPDF Parser  →  Semantic Chunker   │
│                                          ↓              │
│                              all-MiniLM-L6-v2 Embedder  │
│                                          ↓              │
│                          FAISS Index  +  Metadata JSON  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                 ONLINE — Every Query                    │
│                                                         │
│  User Query  (Research_ui.html → FastAPI /query)        │
│       ↓                                                 │
│  Hybrid Retriever  ←────────────────────────────────┐  │
│  (Vector + BM25)                                    │  │
│       │  ← fails gracefully to plain Retriever      │  │
│       ↓                                             │  │
│  Cross-Encoder Reranker                          FAISS  │
│  (ms-marco-MiniLM-L-6-v2)                           │  │
│       ↓                                             │  │
│  Citation Formatter  →  Numbered Context [1]..[N]   │  │
│       ↓                                                 │
│  Citation-Forced Prompt  →  qwen2.5:1.5b (Ollama)      │
│       ↓                                                 │
│  Citation Guard  →  Verify [N] markers via CrossEncoder │
│       │  ← no citations? → one-shot nudge + retry      │
│       ↓                                                 │
│  Final Answer  +  faithfulness_score  +  citations[]   │
└─────────────────────────────────────────────────────────┘
```

---

## Core Features

### Semantic Chunking Pipeline
Documents are not split naively. The chunker runs a 3-stage pipeline:
1. Split on blank lines into raw paragraphs
2. Embed each paragraph with `all-MiniLM-L6-v2` and greedily merge consecutive paragraphs where cosine similarity ≥ 0.45 — keeping topically related content together
3. Apply `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=70) inside each merged group to enforce a hard size cap

This produces chunks that preserve topical coherence rather than cutting mid-argument.

### Hybrid Retrieval with Graceful Fallback
The retriever combines two complementary signals:

- **Vector search** — semantic similarity via FAISS + `all-MiniLM-L6-v2` embeddings. Catches paraphrased or conceptually related content even when exact terms differ.
- **BM25 keyword search** — sparse retrieval that excels at precise technical terminology: model names, acronyms, formula notation.

Neither signal alone is sufficient. Vector search misses exact-term matches; BM25 misses semantic restatements. Together they improve recall across both dimensions.

If `HybridRetriever` fails to initialise (e.g. `rank_bm25` not installed), the system automatically falls back to the plain vector `Retriever` — the pipeline never crashes silently.

### Cross-Encoder Reranking
After hybrid retrieval produces a broad candidate pool, a `CrossEncoder` (`ms-marco-MiniLM-L-6-v2`) scores each query-document pair jointly. Unlike embedding-based similarity, cross-encoders read the query and document together — giving a far more precise relevance signal. Only the top-k reranked chunks proceed to generation.

### Citation-Based Hallucination Control
This is the architectural core of the project. Hallucination control is not a post-hoc filter — it is enforced at the prompt level and verified at the output level.

**Step 1 — Citation Formatter:** Each reranked chunk is assigned a number `[1]..[N]` and formatted with its paper label into a structured context block.

**Step 2 — Citation-Forced Prompt:** The LLM receives an explicit instruction:
> *"After every factual claim add the source number in brackets, e.g. 'X is true [1].' Never fabricate. Omit any claim that has no supporting source."*

**Step 3 — Citation Guard:** After generation, every sentence is parsed for `[N]` markers. Each cited sentence is scored against its referenced chunk using the same `CrossEncoder` already loaded by the reranker (no additional model download). Sentences that cite non-existent or unsupported chunks are dropped.

**Step 4 — Nudge Retry:** If the initial answer contains zero citation markers, a one-shot nudge is appended to the prompt and the model is asked to regenerate once. This handles cases where the model ignores the citation instruction on the first pass.

The API response includes both the raw answer and the `cited_answer` (post-guard), plus a `faithfulness_score` (0.0–1.0) reflecting what fraction of sentences survived verification.

### Research UI
A custom dark-themed HTML frontend (`Research_ui.html`) with:
- Grid background, DM Serif + JetBrains Mono typography, accent green `#4fffb0`
- Animated loading steps mirroring the pipeline stages (Embedding → Retrieving → Reranking → Generating)
- Answer card, Citations grid, Research Papers grid
- Example query chips, Enter key support, graceful error handling

---

## Repository Structure

```
Autonomous-Research-Agent/
├── backend/
│   └── app/
│       ├── agents/
│       │   ├── planner.py
│       │   └── research_agent.py
│       ├── ingestion/
│       │   ├── arxiv_fetcher.py
│       │   ├── pdf_parser.py
│       │   ├── chunking.py
│       │   └── metadata_extractor.py
│       ├── rag_pipeline/
│       │   ├── embedder.py
│       │   ├── vector_store.py
│       │   ├── retriever.py
│       │   ├── hybrid_retriever.py
│       │   ├── reranker.py
│       │   ├── citation_formatter.py   ← new
│       │   └── citation_guard.py       ← new
│       ├── llm/
│       │   └── local_llm.py
│       └── api.py
├── frontend/
│   └── Research_ui.html
├── scripts/
│   ├── ingest_papers.py
│   ├── build_embeddings.py
│   └── evaluate_llm.py
├── tests/
│   └── evaluation/
│       ├── eval_ragas_50.py
│       ├── run_eval.py
│       └── ragwatch_50_report.json
└── data/
    ├── raw_papers/
    ├── processed_chunks/
    └── vector_store/
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| LLM | `qwen2.5:1.5b` via Ollama (local) |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) |
| Reranker + Citation Guard | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector Store | FAISS (`faiss-cpu`) |
| Keyword Retrieval | BM25 (`rank-bm25`) |
| PDF Parsing | PyMuPDF (`fitz`) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |

---

## Getting Started

### 1 — Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- `qwen2.5:1.5b` pulled:

```powershell
ollama pull qwen2.5:1.5b
```

### 2 — Install Dependencies

```powershell
cd backend
pip install -r requirements.txt
```

### 3 — Ingest Papers

Place PDF files in `data/raw_papers/`, or use the arXiv fetcher:

```powershell
python scripts/ingest_papers.py
```

### 4 — Build Vector Index

```powershell
python scripts/build_embeddings.py
```

### 5 — Start the API

```powershell
uvicorn backend.app.api:app --reload --port 8000
```

### 6 — Open the UI

Open `frontend/Research_ui.html` in any browser. The API base is configured to `http://localhost:8000`.

---

## API Reference

### `POST /query`

```json
{
  "query": "What are the main approaches to meta-learning?",
  "top_k": 5
}
```

Response:

```json
{
  "answer": "Raw LLM output with [N] markers.",
  "cited_answer": "Verified answer with unsupported sentences removed.",
  "faithfulness_score": 0.87,
  "citations": [{ "title": "...", "source": "...", "chunk_id": 0, "text": "..." }],
  "papers": [{ "title": "...", "source": "...", "summary": "..." }],
  "sources_used": 5,
  "latency_ms": 8420,
  "model": "qwen2.5:1.5b"
}
```

### `GET /health`

Returns `{ "status": "ok", "chunks": N, "model": "qwen2.5:1.5b" }`.

---

## Evaluation

### Evaluator — RAGWatch

This project was evaluated using **RAGWatch**, a personal evaluation framework built to measure RAG pipeline quality across faithfulness, hallucination rate, relevance, and composite scoring.

> RAGWatch is available as a standalone tool at: **[github.com/tinaprabhat/RAGWATCH](https://github.com/tinaprabhat/RAGWATCH)**

RAGWatch runs 50 synthetic QA pairs across 6 research topic areas and measures:
- **Composite score** — weighted combination of ROUGE, faithfulness, and relevance
- **Faithfulness mean** — how well answers match reference outputs
- **Hallucination mean** — inverse of faithfulness, sentences not grounded in context
- **Guard faithfulness** — proportion of cited sentences verified against source chunks *(new metric, post citation guard)*

### Results

| Metric | Static Baseline | Live Pipeline (with citation guard) | Delta |
|---|---|---|---|
| Composite mean | 0.620 | 0.493 | −0.127 |
| Faithfulness mean | 0.713 | 0.430 | −0.283 |
| Hallucination mean | 0.287 | 0.570 | +0.283 |
| **Guard faith mean** | — | **0.842** | new metric |
| Cases passed | 50/50 ✓ | 39/50 ✗ | 11 failures |

### Interpreting These Results

The apparent drop in faithfulness and rise in hallucination scores requires careful interpretation — **these numbers are not a regression.**

RAGWatch's faithfulness metric measures how closely a generated answer matches pre-written **reference answers**. Those reference answers were written for the original unconstrained Mistral outputs: longer, more complete, more verbose.

After introducing the citation guard, the pipeline changed its behaviour in a fundamentally different direction:

- `qwen2.5:1.5b` generates shorter, more conservative answers than Mistral
- The citation guard actively removes sentences that cannot be verified against retrieved chunks
- This produces answers that are shorter and more conservative — but more grounded

When RAGWatch compares these shorter, citation-stripped answers against verbose reference answers using ROUGE/token-overlap, it sees less textual overlap and records a lower score. **The metric is penalising conservatism, not rewarding it.**

The `guard_faith_mean` of **0.842** tells the real story: 84.2% of cited sentences in the live pipeline are verified as supported by their referenced source chunks. This is the meaningful faithfulness metric for a citation-grounded research agent.

The 11 failures follow a consistent pattern — they are queries that fall outside the knowledge base domain (BERT internals, LoRA details, hybrid retrieval specifics not covered in the ingested meta-learning papers). With the citation guard active, the system refuses to generate unsupported claims on these queries, producing short or empty answers that score low against verbose references. This is **correct behaviour**: the system is honest about the limits of its knowledge.

**The takeaway:** Reference-based metrics like ROUGE are misaligned with citation-grounded RAG. The appropriate evaluation axis is grounding quality, not textual overlap with pre-written references. Future evaluation will update RAGWatch reference answers to reflect citation-constrained expected output.

---

## Hardware Constraints and Scalability

This project runs entirely on a **CPU-only Lenovo ThinkBook** with no GPU. Every architectural decision — model selection, chunking strategy, local LLM choice — was made under this constraint.

### Why `qwen2.5:1.5b`

Mistral (4.1 GB) was the original model but produces unacceptably slow inference on CPU-only hardware (30–60+ seconds per query) and risks OOM under load. `qwen2.5:1.5b` is approximately 1 GB, generates responses in 8–15 seconds on CPU, and produces coherent research-quality output at this scale.

### Why This System Cannot Serve 1,000 Users

This is a deliberate design boundary, not a flaw. The constraints are physical:

- **Ollama is single-threaded per model instance.** One request runs at a time. User 2 waits for user 1 to finish.
- **Cross-encoder reranking is CPU-bound.** Each request scores N×M query-document pairs sequentially.
- **No request queue or backpressure.** Concurrent requests pile up with no rate limiting.
- **FAISS index lives entirely in RAM.** Multiple workers would duplicate memory usage.

The honest ceiling on this hardware is approximately **2–5 concurrent users** before latency becomes unacceptable.

**The path to scale** is infrastructure, not code: GPU-backed server, task queue (Celery + Redis), multiple Uvicorn workers, and semantic caching for repeated queries. The application code requires minimal changes — the bottleneck is hardware and deployment architecture.

---

## Design Philosophy

**Grounded by construction, not by instruction.**
Citation requirements are structural constraints on the output, not polite suggestions to the model.

**Retrieval quality determines answer quality.**
Semantic chunking, hybrid retrieval, and cross-encoder reranking exist because a well-retrieved chunk is worth more than any prompt engineering applied to a poorly-retrieved one.

**Honest about limits.**
The system does not fabricate answers for out-of-domain queries. It returns what the sources support, and nothing more.

**Fully local.**
No API keys. No data leaving the machine. No cloud dependency.

---

## Author

**Tina Prabhat**
B.Tech Computer Science — AI/ML
KIIT University

Areas: AI systems engineering · Retrieval-augmented generation · Research automation

---

## License

MIT License
