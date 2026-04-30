from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:

    def __init__(self, vector_store, documents):

        self.vector_store = vector_store
        self.documents = documents

        tokenized_docs = [doc["text"].split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None

    def search(self, query, k=10):
        if not self.documents:
            return []

        # vector results
        vector_results = self.vector_store.search(query, k)

        # keyword results
        scores = self.bm25.get_scores(query.split())

        top_bm25_idx = np.argsort(scores)[::-1][:k]

        bm25_results = [self.documents[i] for i in top_bm25_idx]

        seen, results = set(), []
        for d in vector_results + bm25_results:
            cid = d["chunk_id"]
            if cid not in seen:
                seen.add(cid)
                results.append(d)

        return results[:k]