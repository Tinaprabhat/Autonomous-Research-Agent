"""
llm_verifier.py - entailment check with an LLM: one batched call per answer.

Each (cited passage, claim) pair is judged strictly: the passage must state or directly imply the claim, including every
number, name, qualifier and comparison direction; extra detail, an unstated inference, a merge of facts from different
places, or a contradiction all count as NOT supported.

Caveats (measured, see info.md section 11.5): the verifier is the same model family as the generator, so their mistakes
can be correlated, and there is no independent fallback verifier. On the 4 real human-labelled unsupported sentences it
caught 2 and missed 2 (an unstated inference and a conflation). Anything unparseable returns None: the guard then marks the
result lexical_only and the pipeline shows a warning.
"""
from __future__ import annotations

import json
import logging
import re
import time

log = logging.getLogger(__name__)

MAX_PASSAGE_CHARS = 800
MAX_CLAIMS_PER_CALL = 12
MAX_RATE_LIMIT_WAIT = 15.0   # seconds: wait out a per-MINUTE token limit once; a daily limit is not waited for
VERIFY_MAX_TOKENS = 1500     # output budget INCLUDES hidden reasoning tokens: the pipeline default (400) truncated ~10% of replies

PROMPT = (
    "You are a strict fact-checker. Below are numbered claims, each with the PASSAGE it cites.\n"
    "For each claim decide whether its passage itself states, or directly and unambiguously implies, the claim. "
    "Check every number, every name, every qualifier and the direction of every comparison. "
    "If the claim adds anything the passage does not give, draws an inference the passage does not state, merges facts "
    "from different places, or contradicts the passage, it is NOT supported.\n\n"
    "{items}\n"
    'Reply with exactly one JSON object and nothing else: {{"results": [{{"id": 1, "supported": true}}, ...]}} '
    "with one entry per claim, in order.")

_ENTRY = re.compile(r'"id"\s*:\s*(\d+)\s*,\s*"supported"\s*:\s*(true|false)', re.I)


def _passage(p) -> str:
    """One cited chunk (str), or several (list[str]) joined, as the passage a claim is checked against."""
    if isinstance(p, str):
        return p.strip()[:MAX_PASSAGE_CHARS]
    return "\n---\n".join(t.strip()[:MAX_PASSAGE_CHARS] for t in p)


def build_prompt(pairs) -> str:
    items = "\n".join(f"[{i}] PASSAGE: {_passage(p)}\n    CLAIM: {c.strip()}\n" for i, (p, c) in enumerate(pairs, 1))
    return PROMPT.format(items=items)


def parse_results(text: str, n: int) -> list[bool] | None:
    """Exactly n verdicts in order, or None if the reply is malformed / incomplete / has unknown ids."""
    verdicts: dict[int, bool] = {}
    try:
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else {}
        for e in data.get("results", []):
            verdicts[int(e["id"])] = bool(e["supported"]) if isinstance(e["supported"], bool) else str(e["supported"]).lower() == "true"
    except (ValueError, KeyError, TypeError, AttributeError):
        verdicts = {int(i): v.lower() == "true" for i, v in _ENTRY.findall(text or "")}
    if set(verdicts) != set(range(1, n + 1)):
        return None
    return [verdicts[i] for i in range(1, n + 1)]


class LLMEntailment:
    """verify(pairs) -> list[bool] (True = supported), or None when the LLM could not give a complete answer."""

    name = "llm"

    def __init__(self, llm):
        self.llm = llm
        try:
            import inspect
            self._can_budget = "max_tokens" in inspect.signature(llm.generate).parameters
        except (TypeError, ValueError):
            self._can_budget = False

    def verify(self, pairs) -> list[bool] | None:
        if not pairs:
            return []
        out: list[bool] = []
        for i in range(0, len(pairs), MAX_CLAIMS_PER_CALL):
            batch = pairs[i:i + MAX_CLAIMS_PER_CALL]
            prompt = build_prompt(batch)
            text = None
            for attempt in (1, 2):
                try:
                    text = (self.llm.generate(prompt, max_tokens=VERIFY_MAX_TOKENS) if self._can_budget
                            else self.llm.generate(prompt))
                    break
                except Exception as e:
                    wait = getattr(e, "retry_after", None)
                    if (attempt == 1 and wait is not None and not getattr(e, "daily", False)
                            and wait <= MAX_RATE_LIMIT_WAIT):
                        log.info("LLM verifier rate-limited; waiting %.1fs", wait)
                        time.sleep(wait + 0.5)
                        continue
                    log.warning("LLM verifier unavailable: %s", e)       # caller marks the result lexical_only
                    return None
            if text is None:
                return None
            res = parse_results(text, len(batch))
            if res is None:
                log.warning("LLM verifier returned an unusable reply")
                return None
            out.extend(res)
        return out
