"""
chunking.py — section-aware semantic chunking with full provenance.

Every chunk dict carries:
    chunk_id    f"{arxiv_id}_{version}_{idx}"  (globally unique)
    arxiv_id, version, section, page, char_start, char_end, text
plus any caller metadata (title, authors, category, published_date, ...).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.ingestion.pdf_parser import Segment

MIN_PARA_CHARS = 80    # ignore trivially short paragraphs
MIN_CHUNK_CHARS = 80   # drop tiny fragments after splitting
SKIP_SECTIONS = {"References"}   # bibliography text is retrieval noise
_RESERVED = {"text", "chunk_id", "arxiv_id", "version", "section", "page",
             "char_start", "char_end"}


def _default_embed(texts: list[str]) -> np.ndarray:
    from backend.app.retrieval.embedder import get_embedder
    return np.asarray(get_embedder().embed_documents(texts))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def _merge_semantic_paragraphs(
    paragraphs: list[str],
    similarity_threshold: float = 0.45,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> list[list[int]]:
    """
    Greedily group consecutive paragraphs whose embedding is similar to the
    running centroid of the current group. Returns groups as lists of indices.

    The centroid is maintained as (sum, count) — O(1) per step, and correct
    regardless of duplicate paragraph text.
    """
    if not paragraphs:
        return []
    embeddings = (embed_fn or _default_embed)(paragraphs)

    groups: list[list[int]] = [[0]]
    vec_sum = embeddings[0].astype("float64").copy()
    for i in range(1, len(paragraphs)):
        centroid = vec_sum / len(groups[-1])
        if _cos(embeddings[i], centroid) >= similarity_threshold:
            groups[-1].append(i)
            vec_sum += embeddings[i]
        else:
            groups.append([i])
            vec_sum = embeddings[i].astype("float64").copy()
    return groups


def chunk_segments(
    segments: list[Segment],
    paper: dict | None = None,
    similarity_threshold: float = 0.45,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
) -> list[dict]:
    """
    Chunk provenance-tagged segments.

    paper : metadata merged into every chunk. `arxiv_id` (required for stable
            IDs) and `version` (default "v1") are read from it.
    Grouping never crosses a section boundary or a page-less/with-page change
    of section, so a chunk's `section` is unambiguous.
    """
    paper = dict(paper or {})
    arxiv_id = str(paper.get("arxiv_id") or "unknown")
    version = str(paper.get("version") or "v1")

    segs = [s for s in segments
            if len(s.text) >= MIN_PARA_CHARS and s.section not in SKIP_SECTIONS]
    if not segs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=70, separators=["\n\n", "\n", ". ", " "])

    # Merge only within a run of identical section.
    docs: list[dict] = []
    idx = 0
    run_start = 0
    while run_start < len(segs):
        run_end = run_start
        while run_end + 1 < len(segs) and segs[run_end + 1].section == segs[run_start].section:
            run_end += 1
        run = segs[run_start:run_end + 1]
        groups = _merge_semantic_paragraphs(
            [s.text for s in run], similarity_threshold, embed_fn)

        for g in groups:
            members = [run[i] for i in g]
            # group text + offsets back into the document text
            group_text, spans, pos = "", [], 0
            for m in members:
                if group_text:
                    group_text += "\n\n"
                    pos += 2
                spans.append((pos, pos + len(m.text), m))
                group_text += m.text
                pos += len(m.text)

            cursor = 0
            for piece in splitter.split_text(group_text):
                piece = piece.strip()
                if len(piece) < MIN_CHUNK_CHARS:
                    continue
                at = group_text.find(piece, max(0, cursor - 80))
                if at < 0:
                    at = group_text.find(piece)
                at = max(at, 0)
                cursor = at + len(piece)

                owner = next((m for (a, b, m) in spans if a <= at < b + 2), members[-1])
                base = next((a for (a, b, m) in spans if m is owner), 0)
                char_start = owner.start + max(at - base, 0)

                doc = {
                    "text": piece,
                    "chunk_id": f"{arxiv_id}_{version}_{idx}",
                    "arxiv_id": arxiv_id,
                    "version": version,
                    "section": owner.section,
                    "page": owner.page,
                    "char_start": char_start,
                    "char_end": char_start + len(piece),
                }
                for k, v in paper.items():
                    if k not in _RESERVED:
                        doc[k] = v
                docs.append(doc)
                idx += 1
        run_start = run_end + 1
    return docs


# [DEAD CODE] `chunk_document` has no callers (checked 2026-10-08); commented out, restore if needed.
# def chunk_document(
#     text: str,
#     metadata: dict | None = None,
#     similarity_threshold: float = 0.45,
#     embed_fn: Callable[[list[str]], np.ndarray] | None = None,
# ) -> list[dict]:
#     """Chunk plain text (no page/section info): paragraphs split on blank lines."""
#     segs, cursor = [], 0
#     for para in text.split("\n\n"):
#         p = para.strip()
#         at = text.find(p, cursor) if p else -1
#         if p and at >= 0:
#             segs.append(Segment(text=p, page=None, section="Unknown", start=at, end=at + len(p)))
#             cursor = at + len(p)
#     return chunk_segments(segs, metadata, similarity_threshold, embed_fn)
