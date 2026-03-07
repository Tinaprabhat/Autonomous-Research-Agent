import numpy as np

class Retriever:

    def __init__(self, vector_store, embedder):

        self.vector_store = vector_store
        self.embedder = embedder

    def search(self, query, k=5):

        query_embedding = self.embedder.embed_query(query)

        query_embedding = np.array([query_embedding]).astype("float32")

        distances, indices = self.vector_store.index.search(query_embedding, k)

        results = []

        # retriever.py
        for idx in indices[0]:
            if idx != -1:                                      # guard for FAISS -1
                results.append(self.vector_store.metadata[idx])
        return results