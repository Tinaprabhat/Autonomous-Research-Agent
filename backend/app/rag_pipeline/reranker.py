from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, documents, top_k=5):
        """
        documents: list of str OR list of dict with "text" key.
        Always returns same type as input.
        """
        if not documents:
            return []

        # Normalize to text for scoring
        def get_text(d):
            return d["text"] if isinstance(d, dict) else d

        pairs = [(query, get_text(d)) for d in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, score in ranked[:top_k]]