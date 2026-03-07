from backend.app.llm.local_llm import LocalLLM
from backend.app.rag_pipeline.embedder import Embedder
from backend.app.rag_pipeline.vector_store import VectorStore
from backend.app.rag_pipeline.retriever import Retriever
from backend.app.rag_pipeline.hybrid_retriever import HybridRetriever
from backend.app.rag_pipeline.reranker import Reranker


class ResearchAgent:

    def __init__(self):
        # Load everything internally — no args needed
        store = VectorStore(dim=384)
        store.load()

        embedder = Embedder()

        # Use HybridRetriever — needs documents list from store metadata
        documents = store.metadata  # list of dicts with "text", "source", etc.
        self.retriever = HybridRetriever(store, documents)
        self.reranker  = Reranker()
        self.llm       = LocalLLM(model="mistral", temperature=0.3,
                                  max_tokens=200, timeout=120)

    def run(self, query: str) -> str:
        # 1. Retrieve
        retrieved = self.retriever.search(query, k=20)

        # 2. Rerank
        reranked = self.reranker.rerank(query, retrieved, top_k=5)

        # 3. Extract text safely — works for both dicts and strings
        def get_text(d):
            return d["text"] if isinstance(d, dict) else str(d)

        context = "\n\n".join(get_text(d) for d in reranked[:3])

        # 4. Generate
        prompt = f"""You are a research assistant.
Use the context below to answer the question concisely.

Context:
{context}

Question: {query}
Answer:"""

        return self.llm.generate(prompt)