"""
golden.py — golden-set schema, loading and validation.

Each case:
    id               unique string
    question         str
    answerable       bool   (False => the corpus cannot answer it; correct behaviour is abstention)
    gold_chunk_ids   list[str]   chunks that contain the answer (empty if unanswerable)
    gold_arxiv_ids   list[str]   papers that contain the answer
    reference_answer str | ""    short human-checked answer
    reviewed         bool   (True once a human verified question + gold chunks)
    verification     "none" | "auto"   ("auto": passed machine verification; see build_golden_set)
    regression       bool   (member of the small CI regression subset)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "data", "golden_set.json")


@dataclass
class GoldenCase:
    id: str
    question: str
    answerable: bool = True
    gold_chunk_ids: list[str] = field(default_factory=list)
    gold_arxiv_ids: list[str] = field(default_factory=list)
    reference_answer: str = ""
    reviewed: bool = False        # a human checked it
    regression: bool = False
    verification: str = "none"    # "auto" = machine-verified (LLM cross-check; earlier sets also used an NLI check), no human review
    source_page: int | None = None
    source_section: str | None = None


class GoldenSetError(ValueError):
    pass


def validate(cases: list[GoldenCase]) -> None:
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise GoldenSetError(f"duplicate case ids: {dupes}")
    for c in cases:
        if not c.question.strip():
            raise GoldenSetError(f"{c.id}: empty question")
        if c.answerable and not (c.gold_chunk_ids or c.gold_arxiv_ids):
            raise GoldenSetError(f"{c.id}: answerable case needs gold_chunk_ids or gold_arxiv_ids")
        if not c.answerable and (c.gold_chunk_ids or c.gold_arxiv_ids):
            raise GoldenSetError(f"{c.id}: unanswerable case must have no gold chunks")


def load(path: str | None = None, only_reviewed: bool = False,
         regression_only: bool = False) -> list[GoldenCase]:
    path = path or DEFAULT_PATH
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    cases = [GoldenCase(**r) for r in raw]
    validate(cases)
    if only_reviewed:   # "verified": human-reviewed OR machine-verified
        cases = [c for c in cases if c.reviewed or c.verification != "none"]
    if regression_only:
        cases = [c for c in cases if c.regression]
    return cases


def save(cases: list[GoldenCase], path: str | None = None, backup: bool = True) -> None:
    """Write the golden set, plus a timestamped copy under data/eval_runs/ (it has vanished once)."""
    validate(cases)
    payload = json.dumps([c.__dict__ for c in cases], indent=2, ensure_ascii=False)
    with open(path or DEFAULT_PATH, "w", encoding="utf-8") as f:
        f.write(payload)
    if backup:
        import time
        os.makedirs("data/eval_runs", exist_ok=True)
        with open(f"data/eval_runs/golden_backup_{time.strftime('%Y%m%dT%H%M%S')}.json", "w", encoding="utf-8") as f:
            f.write(payload)
