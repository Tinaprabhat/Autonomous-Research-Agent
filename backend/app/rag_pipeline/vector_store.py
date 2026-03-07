import faiss
import numpy as np
import os
import json

VECTOR_PATH = "data/vector_store"
INDEX_FILE = os.path.join(VECTOR_PATH, "faiss_index.bin")
META_FILE = os.path.join(VECTOR_PATH, "chunk_metadata.json")


class VectorStore:

    def __init__(self, dim):

        os.makedirs(VECTOR_PATH, exist_ok=True)

        self.index = faiss.IndexFlatL2(dim)

        self.metadata = []

        # vector_store.py → add_embeddings: store full chunk dict, not just text
    def add_embeddings(self, embeddings, chunks):   # chunks = list of dicts
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        for chunk in chunks:
            self.metadata.append(chunk)             # store full dict

    def save(self):

        faiss.write_index(self.index, INDEX_FILE)

        with open(META_FILE, "w") as f:
            json.dump(self.metadata, f)

    def load(self):

        self.index = faiss.read_index(INDEX_FILE)

        with open(META_FILE) as f:
            self.metadata = json.load(f)