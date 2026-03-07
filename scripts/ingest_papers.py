import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from tqdm import tqdm
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np

from backend.app.ingestion.chunking import chunk_document
from backend.app.rag_pipeline.embedder import Embedder
from backend.app.rag_pipeline.vector_store import VectorStore


RAW_DIR   = "data/raw_papers"
CHUNK_DIR = "data/processed_chunks"
META_DIR  = "data/metadata"
VECTOR_DIR = "data/vector_store"


def clean_text(text: str) -> str:
    """Normalize raw PDF text."""
    # Fix common PDF ligatures
    text = text.replace("\ufb01", "fi")   # ﬁ
    text = text.replace("\ufb02", "fl")   # ﬂ
    # Normalize whitespace, preserve line breaks
    text = text.replace("\r", "")
    text = text.replace("\t", " ")
    return text.strip()


def extract_text_from_pdf(pdf_path) -> str:
    """Extract full text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    return "".join(page.get_text() for page in doc)


def main():
    os.makedirs(CHUNK_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(VECTOR_DIR, exist_ok=True)

    embedder = Embedder()
    vector_store = VectorStore(dim=384)

    all_metadata = []

    pdf_files = list(Path(RAW_DIR).glob("*.pdf"))
    print(f"\nFound {len(pdf_files)} papers\n")

    for pdf_file in tqdm(pdf_files, desc="Ingesting papers"):

        # ── 1. Extract & clean text ──────────────────────────────────────────
        raw_text = extract_text_from_pdf(pdf_file)
        text = clean_text(raw_text)

        # ── 2. Chunk ─────────────────────────────────────────────────────────
        chunks = chunk_document(text, metadata={"source": str(pdf_file)})

        if not chunks:
            print(f"  [warn] No chunks produced for {pdf_file.name}, skipping.")
            continue

        # ── 3. Batch embed (one encode call per paper, not per chunk) ─────────
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed_documents(chunk_texts)   # shape: (N, 384)

        # ── 4. Add to vector store with full chunk dict as metadata ───────────
        for chunk, emb in zip(chunks, embeddings):
            emb_array = np.array(emb).astype("float32").reshape(1, -1)
            vector_store.add_embeddings(emb_array, [chunk])  # ← list of dict, NOT bare string

        # ── 5. Save processed chunks to disk ─────────────────────────────────
        chunk_path = os.path.join(CHUNK_DIR, pdf_file.stem + ".json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

        all_metadata.extend(chunks)

    # ── 6. Persist vector index ───────────────────────────────────────────────
    print("\nSaving vector index...")
    vector_store.save()

    # ── 7. Persist full metadata ──────────────────────────────────────────────
    meta_path = os.path.join(META_DIR, "papers_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Metadata saved: {meta_path}")
    print(f"Total chunks ingested: {len(all_metadata)}")
    print("\nIngestion complete.")


if __name__ == "__main__":
    main()