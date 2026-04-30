"""
Unit tests for backend/app/rag_pipeline/vector_store.py

VectorStore's default paths are patched so tests stay in tmp_path and
never touch real project data.
"""
import json
import os
import numpy as np
import pytest

import backend.app.rag_pipeline.vector_store as vs_module
from backend.app.rag_pipeline.vector_store import VectorStore


@pytest.fixture
def store(tmp_vector_store_dir, monkeypatch):
    monkeypatch.setattr(vs_module, "VECTOR_PATH", str(tmp_vector_store_dir))
    monkeypatch.setattr(vs_module, "INDEX_FILE", str(tmp_vector_store_dir / "faiss_index.bin"))
    monkeypatch.setattr(vs_module, "META_FILE",  str(tmp_vector_store_dir / "chunk_metadata.json"))
    return VectorStore(dim=384)


# ── add_embeddings ────────────────────────────────────────────────────────────

def test_add_embeddings_increases_index_count(store, sample_embeddings, sample_chunks):
    assert store.index.ntotal == 0
    store.add_embeddings(sample_embeddings, sample_chunks)
    assert store.index.ntotal == len(sample_embeddings)


def test_add_embeddings_populates_metadata(store, sample_embeddings, sample_chunks):
    store.add_embeddings(sample_embeddings, sample_chunks)
    assert len(store.metadata) == len(sample_chunks)


def test_metadata_preserves_chunk_content(store, sample_embeddings, sample_chunks):
    store.add_embeddings(sample_embeddings, sample_chunks)
    for orig, stored in zip(sample_chunks, store.metadata):
        assert stored["text"] == orig["text"]
        assert stored["chunk_id"] == orig["chunk_id"]


def test_add_embeddings_accepts_float32(store, sample_embeddings, sample_chunks):
    emb_f32 = sample_embeddings.astype("float32")
    store.add_embeddings(emb_f32, sample_chunks)
    assert store.index.ntotal == len(sample_chunks)


# ── save / load round-trip ────────────────────────────────────────────────────

def test_save_creates_files(store, sample_embeddings, sample_chunks, tmp_vector_store_dir):
    store.add_embeddings(sample_embeddings, sample_chunks)
    store.save()
    assert (tmp_vector_store_dir / "faiss_index.bin").exists()
    assert (tmp_vector_store_dir / "chunk_metadata.json").exists()


def test_load_restores_index_count(store, sample_embeddings, sample_chunks, tmp_vector_store_dir,
                                   monkeypatch):
    store.add_embeddings(sample_embeddings, sample_chunks)
    store.save()

    store2 = VectorStore(dim=384)
    monkeypatch.setattr(vs_module, "INDEX_FILE", str(tmp_vector_store_dir / "faiss_index.bin"))
    monkeypatch.setattr(vs_module, "META_FILE",  str(tmp_vector_store_dir / "chunk_metadata.json"))
    store2.load()

    assert store2.index.ntotal == len(sample_embeddings)
    assert len(store2.metadata) == len(sample_chunks)


def test_load_restores_metadata_content(store, sample_embeddings, sample_chunks,
                                        tmp_vector_store_dir, monkeypatch):
    store.add_embeddings(sample_embeddings, sample_chunks)
    store.save()

    store2 = VectorStore(dim=384)
    monkeypatch.setattr(vs_module, "INDEX_FILE", str(tmp_vector_store_dir / "faiss_index.bin"))
    monkeypatch.setattr(vs_module, "META_FILE",  str(tmp_vector_store_dir / "chunk_metadata.json"))
    store2.load()

    for orig, restored in zip(sample_chunks, store2.metadata):
        assert restored["chunk_id"] == orig["chunk_id"]


# ── Dimension ────────────────────────────────────────────────────────────────

def test_store_rejects_wrong_dimension(store, sample_chunks, tmp_vector_store_dir):
    wrong_dim = np.random.rand(5, 128).astype("float32")
    with pytest.raises(Exception):
        store.add_embeddings(wrong_dim, sample_chunks[:5])
        # FAISS raises on dimension mismatch when you add after first batch
        second = np.random.rand(5, 384).astype("float32")
        store.add_embeddings(second, sample_chunks[:5])
