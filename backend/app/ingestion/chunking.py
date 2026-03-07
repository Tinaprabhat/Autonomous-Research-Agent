from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Loaded once at module level to avoid repeated overhead
_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _embed(texts: list[str]) -> np.ndarray:
    model = _get_embed_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def _merge_semantic_paragraphs(
    paragraphs: list[str],
    similarity_threshold: float = 0.45,
) -> list[str]:
    """
    Walk through paragraphs in order and greedily merge consecutive ones
    whose cosine similarity exceeds `similarity_threshold`.

    This keeps topically related paragraphs together so the downstream
    character splitter produces coherent chunks rather than cutting mid-topic.
    """
    if not paragraphs:
        return []

    embeddings = _embed(paragraphs)

    groups: list[list[str]] = [[paragraphs[0]]]
    group_vecs: list[np.ndarray] = [embeddings[0]]

    for i in range(1, len(paragraphs)):
        sim = cosine_similarity([embeddings[i]], [group_vecs[-1]])[0][0]
        if sim >= similarity_threshold:
            # Merge into current group
            groups[-1].append(paragraphs[i])
            # Update representative vector as mean of group
            group_vecs[-1] = embeddings[: i + 1][
                [j for j in range(i + 1) if any(paragraphs[j] in g for g in [groups[-1]])]
            ].mean(axis=0)
        else:
            groups.append([paragraphs[i]])
            group_vecs.append(embeddings[i])

    return ["\n\n".join(group) for group in groups]


def chunk_document(
    text: str,
    metadata: dict | None = None,
    similarity_threshold: float = 0.45,
) -> list[dict]:
    """
    Paragraph-based + semantic chunking with recursive character fallback.

    Pipeline
    --------
    1. Split on blank lines → raw paragraphs.
    2. Drop paragraphs that are too short to be meaningful.
    3. Embed paragraphs and greedily merge semantically similar neighbours
       so that topically related content stays together.
    4. Apply RecursiveCharacterTextSplitter (char_size=300, overlap=50) inside
       each merged group to enforce the hard size cap.
    5. Drop any fragments that are still too short, attach metadata, return.

    Parameters
    ----------
    text                : Raw document text.
    metadata            : Optional dict merged into every chunk document.
    similarity_threshold: Cosine-similarity cutoff for paragraph merging
                          (0 = never merge, 1 = always merge).
                          Default 0.45 works well for general prose.
    """
    MIN_PARA_CHARS = 80   # ignore trivially short paragraphs
    MIN_CHUNK_CHARS = 80  # drop tiny fragments after splitting

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=70,
        separators=["\n\n", "\n", ". ", " "],
    )

    # ── 1. Raw paragraph split ───────────────────────────────────────────────
    raw_paragraphs = [p.strip() for p in text.split("\n\n")]
    paragraphs = [p for p in raw_paragraphs if len(p) >= MIN_PARA_CHARS]

    if not paragraphs:
        return []

    # ── 2. Semantic merging ──────────────────────────────────────────────────
    merged_groups = _merge_semantic_paragraphs(paragraphs, similarity_threshold)

    # ── 3. Recursive character splitting + assembly ──────────────────────────
    documents: list[dict] = []
    chunk_id = 0

    for group_text in merged_groups:
        chunks = splitter.split_text(group_text)

        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < MIN_CHUNK_CHARS:
                continue

            doc: dict = {"text": chunk, "chunk_id": chunk_id}
            if metadata:
                doc.update(metadata)

            documents.append(doc)
            chunk_id += 1

    return documents