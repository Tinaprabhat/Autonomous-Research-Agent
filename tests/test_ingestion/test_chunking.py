"""Unit tests for chunking.py: provenance, global IDs, section boundaries, centroid merge."""
import numpy as np

from backend.app.ingestion.chunking import _merge_semantic_paragraphs, chunk_segments
from backend.app.ingestion.pdf_parser import Segment

PARA = ("Self-attention lets every token attend to every other token in the sequence, "
        "which makes long-range dependencies easy to model in transformers.")
PARA2 = ("Graph neural networks aggregate information over neighbouring nodes using "
         "message passing, which suits relational data such as molecules.")
PAPER = {"arxiv_id": "2401.00001", "version": "v2", "title": "T", "category": "cs.LG"}


def seg(text, section="Method", page=1, start=0):
    return Segment(text=text, page=page, section=section, start=start, end=start + len(text))


def test_chunk_ids_are_globally_unique_across_papers(fake_embedder):
    s = [seg(PARA)]
    a = chunk_segments(s, {"arxiv_id": "1111.1111", "version": "v1"}, embed_fn=fake_embedder.embed_documents)
    b = chunk_segments(s, {"arxiv_id": "2222.2222", "version": "v1"}, embed_fn=fake_embedder.embed_documents)
    assert a[0]["chunk_id"] == "1111.1111_v1_0" and b[0]["chunk_id"] == "2222.2222_v1_0"
    assert not {c["chunk_id"] for c in a} & {c["chunk_id"] for c in b}


def test_chunk_id_includes_version(fake_embedder):
    c = chunk_segments([seg(PARA)], PAPER, embed_fn=fake_embedder.embed_documents)
    assert c[0]["chunk_id"] == "2401.00001_v2_0"


def test_every_chunk_has_provenance_fields(fake_embedder):
    chunks = chunk_segments([seg(PARA, page=4)], PAPER, embed_fn=fake_embedder.embed_documents)
    for c in chunks:
        for k in ("arxiv_id", "version", "section", "page", "char_start", "char_end", "text"):
            assert k in c
        assert c["page"] == 4 and c["section"] == "Method"
        assert c["char_end"] - c["char_start"] == len(c["text"])
        assert c["title"] == "T" and c["category"] == "cs.LG"


def test_char_offsets_point_into_document_text(fake_embedder):
    s1 = seg(PARA, "Introduction", 1, 0)
    s2 = seg(PARA2, "Method", 2, len(PARA) + 2)
    full = PARA + "\n\n" + PARA2
    for c in chunk_segments([s1, s2], PAPER, embed_fn=fake_embedder.embed_documents):
        assert full[c["char_start"]:c["char_end"]] == c["text"]


def test_never_merges_across_sections(fake_embedder):
    same_text = PARA
    chunks = chunk_segments([seg(same_text, "Introduction"), seg(same_text, "Method", start=500)],
                            PAPER, embed_fn=fake_embedder.embed_documents)
    assert {c["section"] for c in chunks} == {"Introduction", "Method"}
    assert len(chunks) == 2


def test_references_section_is_skipped(fake_embedder):
    assert chunk_segments([seg(PARA, "References")], PAPER, embed_fn=fake_embedder.embed_documents) == []


def test_caller_metadata_cannot_override_core_keys(fake_embedder):
    evil = dict(PAPER, chunk_id="hacked", text="hacked", page=999)
    c = chunk_segments([seg(PARA)], evil, embed_fn=fake_embedder.embed_documents)[0]
    assert c["chunk_id"] != "hacked" and c["text"] != "hacked" and c["page"] == 1


def test_short_paragraphs_dropped(fake_embedder):
    assert chunk_segments([seg("too short")], PAPER, embed_fn=fake_embedder.embed_documents) == []


# ── centroid maths (regression for the fragile running-mean recompute) ───────

def test_merge_groups_similar_neighbours_and_splits_different_topics(fake_embedder):
    groups = _merge_semantic_paragraphs([PARA, PARA, PARA2], 0.45, fake_embedder.embed_documents)
    assert groups == [[0, 1], [2]]


def test_merge_handles_duplicate_paragraph_text():
    """Old implementation matched members by text equality and double-counted dupes."""
    vecs = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    embed = lambda ps: np.array([vecs[p] for p in ps])
    assert _merge_semantic_paragraphs(["a", "a", "b", "a"], 0.5, embed) == [[0, 1], [2], [3]]


def test_merge_centroid_is_running_mean():
    # v0=(1,0) v1=(.8,.6) v2=(.6,.8). cos(v1,v0)=.8; centroid(v0,v1)=(.9,.3); cos(v2,centroid)=.82
    # but cos(v2,v0)=.6 -> only a true running centroid merges v2 at threshold .75
    embed = lambda ps: np.array([[1, 0], [0.8, 0.6], [0.6, 0.8]], dtype=float)
    assert _merge_semantic_paragraphs(["x", "y", "z"], 0.75, embed) == [[0, 1, 2]]
    assert _merge_semantic_paragraphs(["x", "y", "z"], 0.85, embed) == [[0], [1, 2]]


def test_merge_empty():
    assert _merge_semantic_paragraphs([], 0.45, lambda p: np.zeros((0, 2))) == []
