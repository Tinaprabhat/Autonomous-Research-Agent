"""
api.py — FastAPI backend for Research Agent UI
Place in: backend/app/api.py

Run from project root:
    uvicorn backend.app.api:app --reload --port 8000
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.rag_pipeline.vector_store import VectorStore
from backend.app.rag_pipeline.embedder import Embedder
from backend.app.rag_pipeline.retriever import Retriever
from backend.app.rag_pipeline.hybrid_retriever import HybridRetriever
from backend.app.rag_pipeline.reranker import Reranker
from backend.app.rag_pipeline.citation_formatter import format_context_with_citations
from backend.app.rag_pipeline.citation_guard import check_citations
from backend.app.llm.local_llm import LocalLLM

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Research Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load all components once at startup ──────────────────────────────────────

print("Loading vector store...")
store = VectorStore(dim=384)
store.load()

embedder           = Embedder()
_fallback_retriever = Retriever(store, embedder)
try:
    retriever = HybridRetriever(store, store.metadata)
except Exception as _e:
    print(f"[warn] HybridRetriever init failed ({_e}), using plain Retriever")
    retriever = _fallback_retriever
reranker  = Reranker()
llm       = LocalLLM(model="qwen2.5:1.5b", temperature=0.3, max_tokens=200, timeout=120)

print(f"Ready. {len(store.metadata)} chunks loaded.")

# ── Prompt ────────────────────────────────────────────────────────────────────

RAG_PROMPT = """You are a research assistant. Answer using ONLY the numbered sources below.
After every factual claim add the source number in brackets, e.g. "X is true [1]." or "Y [2][3]."
Be concise — 3 sentences max. Never fabricate. Omit any claim that has no supporting source.

Context:
{context}

Question: {question}
Answer (cite every claim with [N]):"""

# One-shot nudge appended when the first generation contains no [N] markers
_NUDGE_SUFFIX = "\n(You must cite at least one source using [1], [2], etc.)\nAnswer:"

# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Also support old /ask endpoint shape
class AskRequest(BaseModel):
    question: str

class Citation(BaseModel):
    title:    str | None = None
    source:   str | None = None
    chunk_id: int | None = None
    text:     str | None = None

class QueryResponse(BaseModel):
    answer:            str
    cited_answer:      str
    faithfulness_score: float
    citations:         list[Citation]
    papers:            list[dict]
    sources_used:      int
    latency_ms:        int
    model:             str

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_text(chunk) -> str:
    if isinstance(chunk, dict):
        return chunk.get("text", "")
    return str(chunk)

def get_source(chunk) -> str:
    if isinstance(chunk, dict):
        return chunk.get("source", "")
    return ""

def get_chunk_id(chunk):
    if isinstance(chunk, dict):
        return chunk.get("chunk_id")
    return None

def source_to_title(source: str) -> str:
    return os.path.basename(source).replace("_", " ").replace(".pdf", "")

def run_query(question: str, top_k: int = 5) -> QueryResponse:
    """Core query logic shared by both endpoints."""
    t0 = time.time()

    # 1. Retrieve (hybrid with dense-only fallback)
    try:
        chunks = retriever.search(question, k=top_k * 2)
    except Exception as _e:
        print(f"[warn] HybridRetriever.search failed ({_e}), falling back to plain Retriever")
        chunks = _fallback_retriever.search(question, k=top_k * 2)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # 2. Rerank — reranker handles both dicts and strings
    reranked = reranker.rerank(question, chunks, top_k=top_k)

    # 3. Build numbered context block
    context_block, numbered_chunks = format_context_with_citations(reranked)

    # 4. Generate with citation-forcing prompt
    prompt = RAG_PROMPT.format(context=context_block[:3000], question=question)
    answer = llm.generate(prompt)

    # 5. Citation guard — verify every [N] marker against its chunk
    guard = check_citations(answer, numbered_chunks, reranker.model)
    if not guard.has_citations:
        answer = llm.generate(prompt + _NUDGE_SUFFIX)
        guard  = check_citations(answer, numbered_chunks, reranker.model)

    latency_ms = int((time.time() - t0) * 1000)

    # 6. Build citations + papers
    citations    = []
    papers       = []
    seen_sources = set()

    for chunk in reranked:
        text   = get_text(chunk)
        source = get_source(chunk)
        cid    = get_chunk_id(chunk)
        title  = source_to_title(source) if source else "Unknown source"

        citations.append(Citation(
            title    = title,
            source   = source,
            chunk_id = cid,
            text     = text[:200],
        ))

        if source and source not in seen_sources:
            seen_sources.add(source)
            papers.append({
                "title":   title,
                "source":  source,
                "summary": text[:400],
                "authors": "",
            })

    return QueryResponse(
        answer             = answer,
        cited_answer       = guard.cited_answer,
        faithfulness_score = guard.faithfulness_score,
        citations          = citations,
        papers             = papers,
        sources_used       = len(reranked),
        latency_ms         = latency_ms,
        model              = llm.model,
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Research Agent API running", "chunks": len(store.metadata)}

@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(store.metadata), "model": llm.model}

# UI calls this — POST /query
@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return run_query(req.query, req.top_k)

# Legacy endpoint — kept so old scripts still work
@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = run_query(req.question)
    return {"question": req.question, "answer": result.answer}