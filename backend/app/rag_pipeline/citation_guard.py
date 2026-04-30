"""
citation_guard.py — post-generation citation verifier.

Uses the same cross-encoder already loaded by Reranker to score each
(sentence, cited-chunk) pair.  Sentences whose every cited chunk scores
below `support_threshold` are dropped from the final answer.

faithfulness_score = kept_sentences / total_sentences  (0.0 – 1.0)
has_citations      = True if the answer contained at least one [N] marker.
"""

import re
from dataclasses import dataclass, field

import numpy as np

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class GuardResult:
    cited_answer:       str
    faithfulness_score: float
    has_citations:      bool
    dropped_sentences:  list[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def _get_text(chunk) -> str:
    return chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)


# ── Main function ─────────────────────────────────────────────────────────────

def check_citations(
    answer: str,
    chunks: list,
    cross_encoder,
    support_threshold: float = 0.0,
) -> GuardResult:
    """
    Parameters
    ----------
    answer            : Raw LLM output containing [N] inline markers.
    chunks            : The numbered chunks passed to the prompt (index 0 → [1]).
    cross_encoder     : CrossEncoder instance (reranker.model).
    support_threshold : Raw logit cut-off below which a cited chunk is
                        considered unsupportive.  0.0 drops sentences whose
                        cited chunks score in the negative logit range
                        (not relevant by the cross-encoder's judgement).

    Returns
    -------
    GuardResult with verified cited_answer and faithfulness_score.
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return GuardResult(cited_answer=answer, faithfulness_score=0.0,
                           has_citations=False)

    n = len(chunks)
    kept, dropped = [], []
    cited_count = 0

    for sent in sentences:
        refs = [int(m) for m in _CITATION_RE.findall(sent) if 1 <= int(m) <= n]

        if not refs:
            # Uncited sentence — keep without penalising
            kept.append(sent)
            continue

        cited_count += 1
        clean = _CITATION_RE.sub("", sent).strip()
        pairs  = [(clean, _get_text(chunks[r - 1])) for r in refs]
        scores = np.atleast_1d(cross_encoder.predict(pairs))

        if scores.max() >= support_threshold:
            kept.append(sent)
        else:
            dropped.append(sent)

    faithfulness_score = len(kept) / len(sentences) if sentences else 0.0

    return GuardResult(
        cited_answer       = " ".join(kept).strip(),
        faithfulness_score = round(faithfulness_score, 3),
        has_citations      = cited_count > 0,
        dropped_sentences  = dropped,
    )
