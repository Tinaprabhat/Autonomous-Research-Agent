import os
import re

_NEW_ID = re.compile(r"^\d{4}\.\d{4,5}$")
_OLD_ID = re.compile(r"^[a-z-]+(\.[A-Z]{2})?/\d{7}$")


def is_arxiv_id(value) -> bool:
    return bool(value) and bool(_NEW_ID.match(str(value)) or _OLD_ID.match(str(value)))


def arxiv_url(arxiv_id: str | None) -> str | None:
    """Clickable abstract-page link, built from chunk provenance."""
    return f"https://arxiv.org/abs/{arxiv_id}" if is_arxiv_id(arxiv_id) else None


def arxiv_pdf_url(arxiv_id: str | None, version: str | None = None, page: int | None = None) -> str | None:
    if not is_arxiv_id(arxiv_id):
        return None
    v = version if version and str(version).startswith("v") else ""
    url = f"https://arxiv.org/pdf/{arxiv_id}{v}"
    return f"{url}#page={page}" if page else url


def chunk_label(chunk) -> str:
    if not isinstance(chunk, dict):
        return "source"
    title = chunk.get("title")
    if title:
        label = title
    else:
        src = chunk.get("source", "")
        label = os.path.basename(src).replace(".pdf", "") if src else "source"
    bits = []
    if is_arxiv_id(chunk.get("arxiv_id")):
        bits.append(f"arXiv:{chunk['arxiv_id']}")
    if chunk.get("section") and chunk["section"] != "Unknown":
        bits.append(chunk["section"])
    if chunk.get("page"):
        bits.append(f"p.{chunk['page']}")
    return f"{label} — {', '.join(bits)}" if bits else label


def format_context_with_citations(chunks: list) -> tuple[str, list]:
    """
    Numbers each chunk [1]..[N] and returns:
      context_block : str  — numbered context string to embed in the prompt
      numbered      : list — same list (index i → citation [i+1])
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        parts.append(f"[{i}] ({chunk_label(chunk)})\n{text.strip()}")
    return "\n\n".join(parts), chunks
