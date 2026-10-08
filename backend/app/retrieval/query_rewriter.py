"""
query_rewriter.py — one small LLM call to expand / clarify a query before retrieval.

Fail-safe: any error, timeout or implausible output returns the original query,
so the rewriter can never block or break retrieval.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

REWRITE_PROMPT = (
    "Rewrite the user's question as a single, self-contained search query for a "
    "corpus of academic machine-learning papers. Expand abbreviations, add key "
    "technical synonyms, keep it under 40 words. Output ONLY the query text.\n\n"
    "Question: {q}\nQuery:"
)


class QueryRewriter:

    def __init__(self, llm, max_words: int = 60):
        self.llm = llm
        self.max_words = max_words

    def rewrite(self, query: str) -> str:
        if self.llm is None or not getattr(self.llm, "available", True):
            return query
        try:
            out = self.llm.generate(REWRITE_PROMPT.format(q=query)).strip()
        except Exception as e:
            log.warning("query rewrite failed, using original: %s", e)
            return query
        out = out.strip().strip('"').splitlines()[0].strip() if out else ""
        if not out or len(out.split()) > self.max_words:
            return query
        # keep the original wording too so exact terms still match BM25
        return f"{query} {out}" if query.lower() not in out.lower() else out
