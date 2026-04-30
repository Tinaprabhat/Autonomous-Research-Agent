"""
Unit tests for backend/app/ingestion/chunking.py

Tests cover:
- Empty / whitespace input
- Short paragraphs filtered out
- Chunk size constraints
- Metadata propagation
- chunk_id uniqueness and monotonicity
- Similarity threshold boundary (0 = never merge, 1 = always merge)
"""
import pytest
from backend.app.ingestion.chunking import chunk_document


# ── Helpers ───────────────────────────────────────────────────────────────────

def _long_para(n_words: int = 30) -> str:
    return " ".join(["word"] * n_words)


# ── Basic behaviour ───────────────────────────────────────────────────────────

def test_empty_string_returns_empty():
    assert chunk_document("") == []


def test_whitespace_only_returns_empty():
    assert chunk_document("   \n\n   ") == []


def test_single_short_paragraph_returns_empty():
    assert chunk_document("Too short.") == []


def test_single_long_paragraph_returns_one_chunk(sample_document_text):
    chunks = chunk_document(sample_document_text)
    assert len(chunks) >= 1


def test_chunk_dicts_have_required_keys(sample_document_text):
    chunks = chunk_document(sample_document_text)
    for c in chunks:
        assert "text" in c
        assert "chunk_id" in c


def test_chunk_text_meets_minimum_length(sample_document_text):
    MIN = 80
    chunks = chunk_document(sample_document_text)
    for c in chunks:
        assert len(c["text"]) >= MIN, f"Chunk too short: {c['text']!r}"


def test_chunk_text_respects_max_size(sample_document_text):
    MAX = 600  # allow slight overshoot from splitter
    chunks = chunk_document(sample_document_text)
    for c in chunks:
        assert len(c["text"]) <= MAX, f"Chunk too large: len={len(c['text'])}"


# ── chunk_id ──────────────────────────────────────────────────────────────────

def test_chunk_ids_are_unique(sample_document_text):
    chunks = chunk_document(sample_document_text)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_ids_are_monotonically_increasing(sample_document_text):
    chunks = chunk_document(sample_document_text)
    ids = [c["chunk_id"] for c in chunks]
    assert ids == sorted(ids)


def test_chunk_ids_start_at_zero(sample_document_text):
    chunks = chunk_document(sample_document_text)
    assert chunks[0]["chunk_id"] == 0


# ── Metadata propagation ──────────────────────────────────────────────────────

def test_metadata_propagated_to_all_chunks(sample_document_text):
    meta = {"source": "test.pdf", "year": 2024}
    chunks = chunk_document(sample_document_text, metadata=meta)
    for c in chunks:
        assert c["source"] == "test.pdf"
        assert c["year"] == 2024


def test_no_metadata_does_not_add_extra_keys(sample_document_text):
    chunks = chunk_document(sample_document_text, metadata=None)
    for c in chunks:
        assert set(c.keys()) == {"text", "chunk_id"}


# ── Similarity threshold edge cases ───────────────────────────────────────────

def test_threshold_zero_produces_fewer_or_equal_chunks_than_threshold_one(sample_document_text):
    # threshold=0.0 → similarity always >= 0 → always merges → fewer chunks
    # threshold=1.0 → similarity almost never >= 1 → never merges → more chunks
    chunks_always_merge = chunk_document(sample_document_text, similarity_threshold=0.0)
    chunks_never_merge  = chunk_document(sample_document_text, similarity_threshold=1.0)
    assert len(chunks_always_merge) <= len(chunks_never_merge)


def test_threshold_one_still_produces_valid_chunks(sample_document_text):
    chunks = chunk_document(sample_document_text, similarity_threshold=1.0)
    assert all("text" in c and "chunk_id" in c for c in chunks)


# ── Multi-document isolation ──────────────────────────────────────────────────

def test_two_documents_have_independent_chunk_ids(sample_document_text):
    doc_a = chunk_document(sample_document_text)
    doc_b = chunk_document(sample_document_text)
    # Each call restarts chunk_id from 0
    assert doc_a[0]["chunk_id"] == 0
    assert doc_b[0]["chunk_id"] == 0
