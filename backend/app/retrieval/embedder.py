from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_shared: "Embedder | None" = None


class Embedder:

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, query):
        return self.model.encode([query], show_progress_bar=False)[0]


def get_embedder() -> Embedder:
    """Process-wide shared instance (chunker, ingestion and API reuse one model)."""
    global _shared
    if _shared is None:
        _shared = Embedder()
    return _shared
