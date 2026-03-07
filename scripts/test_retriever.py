
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.rag_pipeline.vector_store import VectorStore
from backend.app.rag_pipeline.retriever import Retriever
from backend.app.rag_pipeline.embedder import Embedder

store = VectorStore(dim=384)
store.load()

embedder = Embedder()

retriever = Retriever(store, embedder)

results = retriever.search("meta learning algorithms", k=3)

for r in results:
    print("\n---\n")
    print(r["text"][:300])