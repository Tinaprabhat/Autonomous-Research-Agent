"""
hybrid_retriever.py — dense (Qdrant) + BM25, fused with Reciprocal Rank Fusion.

RRF scores each document by sum over rankers of 1 / (k + rank), k = 60, so only
*ranks* matter — no score-scale calibration between cosine similarity and BM25.

Filters (arxiv_id / category / author / section / date range) are applied
server-side by Qdrant for the dense side. For the BM25 side, Qdrant resolves the
filter to the set of allowed chunk_ids and BM25 ranks are restricted to it.
"""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

RRF_K = 60
_TOKEN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse several ranked id lists. Returns [(id, score)] best-first."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:

    def __init__(self, vector_store, embedder, documents: list[dict] | None = None,
                 rewriter=None, pool: int = 30):
        """
        vector_store : QdrantStore (needs .search(vec, k, filters) and .scroll_all(filters))
        embedder     : object with .embed_query(str)
        documents    : optional pre-loaded chunk dicts for BM25 (default: scroll the store)
        rewriter     : optional QueryRewriter applied before retrieval
        pool         : candidates pulled from each ranker before fusion
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.rewriter = rewriter
        self.pool = pool
        self.refresh(documents)

    def refresh(self, documents: list[dict] | None = None) -> None:
        self.documents = documents if documents is not None else self.vector_store.scroll_all()
        self._by_id = {d["chunk_id"]: d for d in self.documents}
        self._ids = [d["chunk_id"] for d in self.documents]
        tokenized = [_tokenize(d["text"]) for d in self.documents]
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    def _bm25_ranking(self, query: str, filters: dict | None, n: int) -> list[str]:
        if self.bm25 is None:
            return []
        toks = _tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        allowed = None
        if filters:
            allowed = {d["chunk_id"] for d in self.vector_store.scroll_all(filters, fields=["chunk_id"])}
        order = np.argsort(scores)[::-1]
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            cid = self._ids[i]
            if allowed is not None and cid not in allowed:
                continue
            out.append(cid)
            if len(out) >= n:
                break
        return out

    # Single-signal views, used by the ablation study (dense-only / BM25-only).
    def dense_search(self, query: str, k: int = 10, filters: dict | None = None) -> list[dict]:
        return self.vector_store.search(self.embedder.embed_query(query), k, filters=filters)

    def bm25_search(self, query: str, k: int = 10, filters: dict | None = None) -> list[dict]:
        return [dict(self._by_id[c]) for c in self._bm25_ranking(query, filters, k)]

    def search(self, query: str, k: int = 10, filters: dict | None = None,
               rewrite: bool = True) -> list[dict]:
        q = query
        if rewrite and self.rewriter is not None:
            q = self.rewriter.rewrite(query)

        qvec = self.embedder.embed_query(q)
        n = max(self.pool, k)
        dense = self.vector_store.search(qvec, n, filters=filters)
        for d in dense:
            self._by_id.setdefault(d["chunk_id"], d)
        dense_ids = [d["chunk_id"] for d in dense]
        bm25_ids = self._bm25_ranking(q, filters, n)

        fused = rrf_fuse([dense_ids, bm25_ids])
        results = []
        for cid, score in fused[:k]:
            doc = dict(self._by_id[cid])
            doc["rrf_score"] = round(score, 6)
            results.append(doc)
        return results
