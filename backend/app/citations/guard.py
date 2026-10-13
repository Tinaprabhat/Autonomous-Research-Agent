"""
citation_guard.py — post-generation verifier using real NLI entailment.

For each answer sentence:
  * no valid [N] marker            -> dropped (uncited claims are not allowed)
  * cited chunk(s) do not ENTAIL it -> dropped (P(entailment) < threshold)
  * otherwise                       -> kept

There is no raw-answer fallback: if everything is dropped the caller gets an
empty `cited_answer`, and if more than half the sentences are dropped
`abstain` is True so the pipeline can answer "not enough evidence" instead of
forcing an answer.

faithfulness_score = kept_sentences / total_sentences  (0.0 – 1.0)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from backend.app import config

_CITATION_RE = re.compile(r"\[(\d+)\]")
_ABBREV = ["et al.", "e.g.", "i.e.", "Fig.", "Figs.", "Eq.", "Sec.", "vs.", "cf.", "No.", "approx.", "Dr.", "Prof.", "Mr.", "Ms.", "Ref.", "Refs.", "Tab."]
ABSTAIN_DROP_RATIO = 0.5
_INITIALS_END = re.compile(r"(?:^|[\s(])(?:[A-Za-z]\.){2,}$|(?:^|[\s(])[A-Z]\.$")    # "J.S." / "U.S." / "J."
_SENTENCE_START = re.compile(r"[A-Z0-9\"'“‘(\[]")                                    # a new sentence starts like this


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class GuardResult:
    cited_answer:       str
    faithfulness_score: float
    has_citations:      bool
    dropped_sentences:  list[str] = field(default_factory=list)
    kept_sentences:     list[str] = field(default_factory=list)
    abstain:            bool = False


# ── NLI model wrapper ─────────────────────────────────────────────────────────

class NLIVerifier:
    """cross-encoder NLI model; `.entailment(premise, hypothesis)` -> P(entails)."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name or config.NLI_MODEL
        # 3-sentence premise windows + one hypothesis fit well inside 256 tokens; the default 512 only adds padding cost
        self.model = CrossEncoder(self.model_name, max_length=256)
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self._ent_idx = next(i for i, l in id2label.items() if l.startswith("entail"))

    MAX_WINDOWS = 3     # NLI is the latency bottleneck: only the windows most similar to the claim are scored

    @staticmethod
    def _tok(text: str) -> set[str]:
        return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}

    @classmethod
    def _top_windows(cls, premise: str, hyp: str) -> list[str]:
        wins = cls._windows(premise)
        if len(wins) <= cls.MAX_WINDOWS:
            return wins
        h = cls._tok(hyp)
        scored = sorted(wins, key=lambda w: len(h & cls._tok(w)) / (len(cls._tok(w)) ** 0.5 or 1), reverse=True)
        return scored[:cls.MAX_WINDOWS]

    @staticmethod
    def _windows(text: str, size: int = 3) -> list[str]:
        """Sliding sentence windows: NLI models handle short premises far better than 500-char blobs."""
        sents = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
        if len(sents) <= size:
            return [text]
        return [" ".join(sents[i:i + size]) for i in range(len(sents) - size + 1)]

    def entailment_batch(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """P(premise entails hypothesis) per pair = max over sentence windows of the premise."""
        if not pairs:
            return np.zeros(0)
        flat, owner = [], []
        for i, (premise, hyp) in enumerate(pairs):
            for w in self._top_windows(premise, hyp):
                flat.append((w, hyp))
                owner.append(i)
        logits = np.asarray(self.model.predict(flat, batch_size=32), dtype="float64")
        logits = logits.reshape(len(flat), -1)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        ent = (e / e.sum(axis=1, keepdims=True))[:, self._ent_idx]
        out = np.zeros(len(pairs))
        for k, i in enumerate(owner):
            out[i] = max(out[i], ent[k])
        return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    t = text.strip()
    if not t:
        return []
    # "claim. [1]" -> "claim [1]."  so a marker stays with its sentence
    t = re.sub(r"([.!?])\s*((?:\[\d+\])+)", r" \2\1", t)
    for i, a in enumerate(_ABBREV):
        t = t.replace(a, a.replace(".", f"․"))   # protect with one-dot leader
    parts = [p.replace("․", ".").strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    # A period is not always a sentence end: re-join "J.S. | Bach", "augm. | + CL)", "Fig. | a" style splits.
    merged: list[str] = []
    for p in parts:
        if merged and (_INITIALS_END.search(merged[-1]) or not _SENTENCE_START.match(p)):
            merged[-1] += " " + p
        else:
            merged.append(p)
    return merged


def _get_text(chunk) -> str:
    return chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)


# ── Main function ─────────────────────────────────────────────────────────────

def check_citations(
    answer: str,
    chunks: list,
    verifier,
    support_threshold: float = 0.5,
) -> GuardResult:
    """
    answer     : raw LLM output with [N] markers (index 0 of `chunks` -> [1]).
    chunks     : the numbered chunks given to the prompt.
    verifier   : object with .entailment_batch(list[(premise, hypothesis)]) -> array
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return GuardResult("", 0.0, False, abstain=True)

    n = len(chunks)
    refs_per_sentence: list[list[int]] = []
    pairs: list[tuple[str, str]] = []
    owner: list[int] = []          # pair index -> sentence index

    for si, sent in enumerate(sentences):
        refs = sorted({int(m) for m in _CITATION_RE.findall(sent) if 1 <= int(m) <= n})
        refs_per_sentence.append(refs)
        hyp = re.sub(r"\s+", " ", _CITATION_RE.sub("", sent)).strip()
        for r in refs:
            pairs.append((_get_text(chunks[r - 1]), hyp))
            owner.append(si)

    probs = verifier.entailment_batch(pairs) if pairs else []
    best = [0.0] * len(sentences)
    for pi, si in enumerate(owner):
        best[si] = max(best[si], float(probs[pi]))

    kept, dropped = [], []
    for si, sent in enumerate(sentences):
        if refs_per_sentence[si] and best[si] >= support_threshold:
            kept.append(sent)
        else:
            dropped.append(sent)

    total = len(sentences)
    drop_ratio = len(dropped) / total
    return GuardResult(
        cited_answer       = " ".join(kept).strip(),
        faithfulness_score = round(len(kept) / total, 3),
        has_citations      = any(refs_per_sentence),
        dropped_sentences  = dropped,
        kept_sentences     = kept,
        abstain            = drop_ratio > ABSTAIN_DROP_RATIO or not kept,
    )
