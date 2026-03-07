import os
import json

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.rag_pipeline.embedder import Embedder
from backend.app.rag_pipeline.vector_store import VectorStore

CHUNK_DIR = "data/processed_chunks"


def load_chunks():

    texts = []

    for file in os.listdir(CHUNK_DIR):

        path = os.path.join(CHUNK_DIR, file)

        with open(path) as f:
            chunks = json.load(f)

            texts.extend(chunks)

    return texts


def build_vector_db():

    texts = load_chunks()

    embedder = Embedder()

    embeddings = embedder.embed_documents(texts)

    dim = len(embeddings[0])

    store = VectorStore(dim)

    store.add_embeddings(embeddings, texts)

    store.save()

    print("Vector database built.")


if __name__ == "__main__":

    build_vector_db()