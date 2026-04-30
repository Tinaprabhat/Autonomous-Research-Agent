import os


def format_context_with_citations(chunks: list) -> tuple[str, list]:
    """
    Numbers each chunk [1]..[N] and returns:
      context_block : str  — numbered context string to embed in the prompt
      numbered      : list — same list (index i → citation [i+1])

    Each block looks like:
        [1] (paper-name)
        <chunk text>
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        text   = chunk.get("text",   "") if isinstance(chunk, dict) else str(chunk)
        source = chunk.get("source", "") if isinstance(chunk, dict) else ""
        label  = os.path.basename(source).replace(".pdf", "") if source else "source"
        parts.append(f"[{i}] ({label})\n{text.strip()}")
    return "\n\n".join(parts), chunks
