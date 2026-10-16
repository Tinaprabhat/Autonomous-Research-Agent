"""
guard.py - post-generation citation verifier: a layered check with graceful degradation.

For each answer sentence, in order:
  0. hedge        no [N] and a "the sources don't say" phrase           -> passed through unchecked
  1. citation     no valid [N] marker                                   -> dropped (uncited claims are not allowed)
  A. screen       (free, deterministic, always on)
       overlap    < GUARD_OVERLAP_THRESHOLD of the sentence's content words occur in the cited chunks -> dropped
       specifics  a number / acronym / name of the sentence is missing from the cited chunks            -> dropped
  B. entailment   (EntailmentChain: the LLM verifier; if it cannot run, nothing)
       does a cited passage state or directly imply the sentence?   no -> dropped, yes -> kept

Layer B is where meaning is checked. If neither verifier can run, the sentences that passed layer A are kept but the
result is marked verification="lexical_only" and the pipeline shows a warning: it is never presented as fully verified.

What this does NOT do (measured, docs/case_study.md section 3.3): on the 4 real human-labelled unsupported sentences, both
entailment verifiers caught the same 2 and missed an unstated inference and a conflation of two facts. Layer A stops
cheap errors (wrong numbers, foreign names, wrong-chunk citations); layer B stops contradictions and added claims.
Subtle over-reach still gets through.

There is no raw-answer fallback: if everything is dropped the caller gets an empty `cited_answer`, and if more than
half of the checkable sentences are dropped `abstain` is True so the pipeline can tag the answer.
faithfulness_score = kept / checkable sentences (passthrough sentences are on neither side).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from backend.app import config

log = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")
_ABBREV = ["et al.", "e.g.", "i.e.", "Fig.", "Figs.", "Eq.", "Sec.", "vs.", "cf.", "No.", "approx.", "Dr.", "Prof.", "Mr.", "Ms.", "Ref.", "Refs.", "Tab."]
ABSTAIN_DROP_RATIO = 0.5
_INITIALS_END = re.compile(r"(?:^|[\s(])(?:[A-Za-z]\.){2,}$|(?:^|[\s(])[A-Z]\.$")    # "J.S." / "U.S." / "J."
_SENTENCE_START = re.compile(r"[A-Z0-9\"'“‘(\[]")                                    # a new sentence starts like this
_ANCHOR_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?%?")

# Hedge: phrases the LLM uses when the sources do not contain the answer. Regex, case-insensitive.
_HEDGE_RE = re.compile(
    r"\b(?:"
    r"not\s+sure|"
    r"(?:could(?:\s*n['’]?t|\s+not)|cannot|can['’]?t|unable\s+to)\s+(?:find|locate|confirm|determine|verify)|"
    r"no\s+(?:relevant\s+)?(?:information|mention|evidence|details?)\b|"
    r"(?:not|never)\s+(?:mentioned|stated|specified|covered|described|discussed|addressed|reported|provided)|"
    r"(?:do(?:es)?\s+not|don['’]?t|doesn['’]?t)\s+(?:contain|mention|state|specify|say|provide|include|address)|"
    r"(?:no|nothing)\s+(?:in|within)\s+the\s+(?:provided\s+)?(?:sources?|passages?|context)|"
    r"insufficient\s+(?:information|evidence)|"
    r"none\s+of\s+the\s+(?:provided\s+|given\s+|retrieved\s+)?(?:sources?|passages?|documents?|excerpts?|context)|"
    r"(?:sources?|passages?|documents?|excerpts?)\s+(?:provided\s+)?(?:do(?:es)?\s+not|don['’]?t|doesn['’]?t)"
    r")",
    re.I)
HEDGE_MAX_CHARS = 240       # a long sentence that merely contains a hedge phrase is still checked


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class GuardResult:
    cited_answer:          str
    faithfulness_score:    float
    has_citations:         bool
    dropped_sentences:     list[str] = field(default_factory=list)
    kept_sentences:        list[str] = field(default_factory=list)
    abstain:               bool = False
    passthrough_sentences: list[str] = field(default_factory=list)
    details:               list[dict] = field(default_factory=list)   # per sentence: decision, reason, scores
    verification:          str = ""     # llm (entailment-checked) | lexical_only (the LLM verifier could not run or is off)


# ── Layer A helpers ───────────────────────────────────────────────────────────

def _tok(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}


def lexical_overlap(hypothesis: str, chunk_text: str) -> float:
    """Share of the sentence's content words (len > 2) that appear in the chunk: 0..1."""
    h = _tok(hypothesis)
    return len(h & _tok(chunk_text)) / len(h) if h else 0.0


def anchors(sentence: str) -> list[str]:
    """The checkable specifics: numbers, acronyms, mixed-case words and non-initial capitalised words (lower-cased)."""
    out = [m.group(0).replace(",", "").lower() for m in _ANCHOR_NUM.finditer(sentence)]
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", sentence)
    for i, w in enumerate(words):
        core = w.strip("-")
        if len(core) < 2:
            continue
        if core.isupper() or core[1:] != core[1:].lower() or (core[0].isupper() and i > 0):
            out.append(core.lower())
    return list(dict.fromkeys(out))


def missing_anchors(sentence: str, chunk_texts: list[str]) -> tuple[float, list[str]]:
    """(share of anchors found in the cited chunks taken together, the anchors that are missing). No anchors -> (1.0, [])."""
    a = anchors(sentence)
    if not a:
        return 1.0, []
    low = " ".join(chunk_texts).lower().replace(",", "")
    miss = [x for x in a if x not in low]
    return 1 - len(miss) / len(a), miss


def is_hedge(sentence: str) -> bool:
    return bool(_HEDGE_RE.search(sentence)) and len(sentence) <= HEDGE_MAX_CHARS


def screen_claim(claim: str, chunk_texts: list[str], overlap_threshold: float | None = None,
                 specifics_min: float | None = None) -> dict:
    """Layer A for one claim against the chunks it cites. {"ok": bool, "reason": str (if not ok), "overlap", "specifics", "missing"?}"""
    ov_t = config.GUARD_OVERLAP_THRESHOLD if overlap_threshold is None else overlap_threshold
    sp_t = config.GUARD_SPECIFICS_MIN if specifics_min is None else specifics_min
    ov = max(lexical_overlap(claim, t) for t in chunk_texts)
    cover, miss = missing_anchors(claim, chunk_texts)
    info = {"overlap": round(ov, 3), "specifics": round(cover, 3)}
    if ov < ov_t:
        return {"ok": False, "reason": "low_overlap", **info}
    if cover < sp_t:
        return {"ok": False, "reason": "missing_specifics", "missing": miss, **info}
    return {"ok": True, **info}


# ── Layer B: entailment chain ─────────────────────────────────────────────────

class EntailmentChain:
    """
    Runs the LLM verifier (layer B) and returns (flags, source). The verifier has
    verify(items) -> list[bool] | None, where items = [(cited_chunk_texts, claim)] and True means supported.
    None / an exception = "could not decide": there is deliberately NO second verifier, so the source is "none" and
    every flag is True; the caller then marks the result lexical_only and the pipeline shows a warning.
    GUARD_ENTAILMENT = llm (default) | off (layer A only).
    """

    def __init__(self, llm=None, mode: str | None = None):
        self.llm = llm
        self.mode = (mode or config.GUARD_ENTAILMENT).lower()
        if self.mode not in ("llm", "off"):
            log.warning("GUARD_ENTAILMENT=%r is not supported (use llm or off); using llm", self.mode)
            self.mode = "llm"

    def verify(self, items):
        if not items or self.mode == "off" or self.llm is None:
            return [True] * len(items), "none"
        try:
            res = self.llm.verify(items)
        except Exception as e:                               # a verifier must never take the answer down
            log.warning("entailment verifier %s failed: %s", getattr(self.llm, "name", self.llm), e)
            return [True] * len(items), "none"
        if res is not None and len(res) == len(items):
            return list(res), getattr(self.llm, "name", "verifier")
        return [True] * len(items), "none"


# ── Sentence splitting ────────────────────────────────────────────────────────

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
    verifier=None,
    overlap_threshold: float | None = None,
    specifics_min: float | None = None,
) -> GuardResult:
    """
    answer     : raw LLM output with [N] markers (index 0 of `chunks` -> [1]).
    chunks     : the numbered chunks given to the prompt.
    verifier   : EntailmentChain-like object with .verify([(cited_chunk_texts, claim)]) -> (flags, source);
                 None runs layer A only.
    """
    ov_t = config.GUARD_OVERLAP_THRESHOLD if overlap_threshold is None else overlap_threshold
    sp_t = config.GUARD_SPECIFICS_MIN if specifics_min is None else specifics_min
    sentences = _split_sentences(answer)
    if not sentences:
        return GuardResult("", 0.0, False, abstain=True)

    n = len(chunks)
    decisions: list[dict] = [{} for _ in sentences]
    refs_per_sentence: list[list[int]] = []
    cand: list[tuple[int, list[str], str]] = []                  # (sentence index, cited chunk texts, claim)

    for si, sent in enumerate(sentences):
        refs = sorted({int(m) for m in _CITATION_RE.findall(sent) if 1 <= int(m) <= n})
        refs_per_sentence.append(refs)
        hyp = re.sub(r"\s+", " ", _CITATION_RE.sub("", sent)).strip()
        if not refs:
            decisions[si] = ({"decision": "passthrough", "reason": "hedge"} if is_hedge(sent)
                             else {"decision": "dropped", "reason": "uncited"})
            continue
        texts = [_get_text(chunks[r - 1]) for r in refs]
        scr = screen_claim(hyp, texts, ov_t, sp_t)
        if scr["ok"]:
            decisions[si] = {"overlap": scr["overlap"], "specifics": scr["specifics"]}
            cand.append((si, texts, hyp))
        else:
            decisions[si] = {"decision": "dropped", **{k: v for k, v in scr.items() if k != "ok"}}

    source = ""
    if cand:                                                      # layer B only for what layer A let through
        if verifier is None:
            flags, source = [True] * len(cand), "none"
        else:
            flags, source = verifier.verify([(t, h) for _, t, h in cand])
        for (si, _, _), ok in zip(cand, flags):
            decisions[si] = {**decisions[si], "decision": "kept" if ok else "dropped",
                             **({} if ok else {"reason": "not_entailed"}), "verifier": source}

    kept, dropped, passthrough, shown, details = [], [], [], [], []
    for si, sent in enumerate(sentences):
        d = decisions[si]
        details.append({"sentence": sent, **d})
        if d["decision"] == "kept":
            kept.append(sent)
            shown.append(sent)
        elif d["decision"] == "passthrough":
            passthrough.append(sent)
            shown.append(sent)
        else:
            dropped.append(sent)

    checkable = len(kept) + len(dropped)
    if checkable == 0:                      # only hedges: an honest "the sources don't say" answer
        faith, abstain = 1.0, False
    else:
        faith = round(len(kept) / checkable, 3)
        abstain = len(dropped) / checkable > ABSTAIN_DROP_RATIO or not kept
    return GuardResult(
        cited_answer          = " ".join(shown).strip(),
        faithfulness_score    = faith,
        has_citations         = any(refs_per_sentence),
        dropped_sentences     = dropped,
        kept_sentences        = kept,
        abstain               = abstain,
        passthrough_sentences = passthrough,
        details               = details,
        verification          = "lexical_only" if source == "none" else source,
    )
