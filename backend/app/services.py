"""
services.py — wires the concrete components into a ResearchPipeline.
"""
from __future__ import annotations

import logging

from backend.app.safety.guardrails import JailbreakDetector, OutputGuard
from backend.app.llm.router import LLMRouter, cheap_rewrite_llm
from backend.app.pipeline import ResearchPipeline
from backend.app.citations.guard import EntailmentChain
from backend.app.citations.llm_verifier import LLMEntailment
from backend.app.retrieval.embedder import get_embedder
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.query_rewriter import QueryRewriter
from backend.app.retrieval.reranker import Reranker
from backend.app.storage.qdrant_store import QdrantStore

log = logging.getLogger(__name__)


def build_pipeline(store: QdrantStore | None = None, use_rewriter: bool = True):
    """Returns (pipeline, store). Heavy: loads the embedder, reranker and jailbreak models."""
    store = store or QdrantStore()
    store.verify_indexes() if store.count() else None
    embedder = get_embedder()
    rewriter = QueryRewriter(cheap_rewrite_llm()) if use_rewriter else None
    retriever = HybridRetriever(store, embedder, rewriter=rewriter)
    llm = LLMRouter()
    pipeline = ResearchPipeline(
        retriever=retriever,
        reranker=Reranker(),
        verifier=EntailmentChain(llm=LLMEntailment(llm)),
        llm=llm,
        input_guard=JailbreakDetector(),
        output_guard=OutputGuard(),
    )
    log.info("pipeline ready: %d chunks", len(retriever.documents))
    return pipeline, store
