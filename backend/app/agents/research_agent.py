from backend.app.llm.local_llm import LocalLLM
from backend.app.rag_pipeline.embedder import Embedder
from backend.app.rag_pipeline.vector_store import VectorStore
from backend.app.rag_pipeline.retriever import Retriever
from backend.app.rag_pipeline.hybrid_retriever import HybridRetriever
from backend.app.rag_pipeline.reranker import Reranker
from backend.app.rag_pipeline.citation_formatter import format_context_with_citations
from backend.app.rag_pipeline.citation_guard import check_citations


class ResearchAgent:

    def __init__(self):
        # Load everything internally — no args needed
        store = VectorStore(dim=384)
        store.load()

        embedder = Embedder()
        self._fallback_retriever = Retriever(store, embedder)

        documents = store.metadata
        try:
            self.retriever = HybridRetriever(store, documents)
        except Exception as _e:
            print(f"[warn] HybridRetriever init failed ({_e}), using plain Retriever")
            self.retriever = self._fallback_retriever
        self.reranker  = Reranker()
        self.llm       = LocalLLM(model="qwen2.5:1.5b", temperature=0.3,
                                  max_tokens=200, timeout=120)

    def run(self, query: str) -> str:
        # 1. Retrieve (hybrid with dense-only fallback)
        try:
            retrieved = self.retriever.search(query, k=20)
        except Exception as _e:
            print(f"[warn] HybridRetriever.search failed ({_e}), falling back to plain Retriever")
            retrieved = self._fallback_retriever.search(query, k=20)

        # 2. Rerank
        reranked = self.reranker.rerank(query, retrieved, top_k=5)

        # 3. Build numbered citation context
        context, numbered_chunks = format_context_with_citations(reranked)

        # 4. Generate with citation prompt
        prompt = (
            "You are a research assistant. Answer using ONLY the numbered sources below.\n"
            "After every factual claim add the source number in brackets, e.g. \"X is true [1].\" or \"Y [2][3].\"\n"
            "Be concise — 3 sentences max. Never fabricate. Omit any claim that has no supporting source.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer (cite every claim with [N]):"
        )

        answer = self.llm.generate(prompt)

        # 5. Nudge if model produced no citations
        if not any(f"[{i}]" in answer for i in range(1, len(numbered_chunks) + 1)):
            answer = self.llm.generate(
                prompt + "\n(You must cite at least one source using [1], [2], etc.)\nAnswer:"
            )

        # 6. Verify citations with cross-encoder guard
        guard = check_citations(answer, numbered_chunks, self.reranker.model)
        return guard.cited_answer or answer