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
from backend.app.rag_pipeline.reranker import Reranker
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

embedder  = Embedder()
retriever = Retriever(store, embedder)
reranker  = Reranker()
llm       = LocalLLM(model="mistral", temperature=0.3, max_tokens=200, timeout=120)

print(f"Ready. {len(store.metadata)} chunks loaded.")

# ── Prompt ────────────────────────────────────────────────────────────────────

RAG_PROMPT = """Answer the question using ONLY the context below.
Be concise — 3 sentences max. Do not fabricate.
If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}
Answer:"""

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
    answer:       str
    citations:    list[Citation]
    papers:       list[dict]
    sources_used: int
    latency_ms:   int
    model:        str

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

    # 1. Retrieve
    chunks = retriever.search(question, k=top_k * 2)
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # 2. Rerank — reranker handles both dicts and strings
    reranked = reranker.rerank(question, chunks, top_k=top_k)

    # 3. Build context
    context = "\n\n".join(get_text(c) for c in reranked)

    # 4. Generate
    prompt = RAG_PROMPT.format(context=context[:3000], question=question)
    answer = llm.generate(prompt)

    latency_ms = int((time.time() - t0) * 1000)

    # 5. Build citations + papers
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
        answer       = answer,
        citations    = citations,
        papers       = papers,
        sources_used = len(reranked),
        latency_ms   = latency_ms,
        model        = llm.model,
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