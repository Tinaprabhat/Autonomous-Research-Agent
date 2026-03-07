
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.rag_pipeline.vector_store import VectorStore

store = VectorStore(dim=384)

store.load()

print("Vector DB loaded successfully")

print("Total documents:", len(store.metadata))