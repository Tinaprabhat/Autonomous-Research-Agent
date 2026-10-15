"""
ragas_metrics.py - RAGAS generation metrics behind the interface the evaluation loop already uses (replaces DeepEval).

Why DeepEval was dropped (info.md section 11.8): its faithfulness metric scored every answer 1.0 in our runs. Two causes:
(1) by default it counts "borderline" claims as faithful (`penalize_ambiguous_claims=False`) and (2) with the Groq judge it
marked every claim of 3 of the 5 human-labelled unfaithful answers "yes", and it changed its mind between identical runs.
Setting (1) is a one-line fix but (2) is judge leniency, so the metric was replaced.

Metrics (all LLM-judged by the Groq model passed to RagasJudge; default openai/gpt-oss-120b at its DEFAULT reasoning effort:
at "low" effort the judge missed the unsupported claim in the answer used to test it):

  faithfulness   RAGAS Faithfulness: share of the response's statements that the retrieved contexts support
  hallucination  1 - faithfulness: share of statements NOT supported by the contexts (RAGAS ships no separate metric;
                 this is the statement-level complement, and `hallucinated` is True when any statement is unsupported)
  correctness    F1 of RAGAS FactualCorrectness precision (share of the response's claims the reference supports) and
                 coverage (share of the reference's claims the response states)
  completeness   coverage: share of the reference answer's claims the response states. RAGAS's own "recall" is tp/(tp+fn)
                 with tp = response claims the reference supports, so it is 0 whenever no response claim is supported
                 (e.g. a correct answer that restates the question's context, checked against a terse reference); it is
                 computed and kept as `fact_recall` but not used as completeness

Reference answers in the golden set are machine-written (not human-reviewed), so correctness/completeness are relative to them.

RAGAS 0.4.3 does not import in an environment with langchain-community >= 0.4 (it needs `chat_models.vertexai`, which was removed);
`load_ragas()` installs a stub for that one module instead of touching the installed packages.
"""
from __future__ import annotations

import asyncio
import re
import sys
# [DEAD CODE] unused import (checked 2026-10-15)
# import time
import types
from dataclasses import dataclass

from backend.app import config

DEFAULT_JUDGE = "openai/gpt-oss-120b"
MAX_TOKENS = 4096          # output budget INCLUDES hidden reasoning tokens; smaller values truncate structured replies


class JudgeBudgetExhausted(RuntimeError):
    """The judge model's daily quota is used up (a per-minute limit is waited out instead)."""


@dataclass
class EvalCase:
    question: str
    response: str
    contexts: list[str]
    reference: str = ""


def load_ragas():
    """Import ragas (with the compatibility stub if needed) and return (llm_factory, metric classes)."""
    key = "langchain_community.chat_models.vertexai"
    if key not in sys.modules:
        try:
            __import__(key)
        except Exception:
            stub = types.ModuleType(key)
            stub.ChatVertexAI = type("ChatVertexAI", (), {})
            sys.modules[key] = stub
    from ragas.llms import llm_factory
    from ragas.metrics.collections import FactualCorrectness, Faithfulness
    return llm_factory, Faithfulness, FactualCorrectness


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def is_daily_budget_error(e: BaseException) -> bool:
    """True if `e` or anything in its cause chain is a per-DAY quota failure (RAGAS/instructor wrap the original 429)."""
    seen = 0
    while e is not None and seen < 10:
        if isinstance(e, JudgeBudgetExhausted) or "tokens per day" in str(e) or "(TPD)" in str(e) or "requests per day" in str(e):
            return True
        e = e.__cause__ or e.__context__
        seen += 1
    return False


def _wait_seconds(message: str) -> float:
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
    return (float(m.group(2)) + 60 * int(m.group(1) or 0) + 1.0) if m else 20.0


class RagasJudge:
    """Owns the Groq-backed RAGAS LLM; computes (and caches per case) the raw RAGAS scores."""

    def __init__(self, model: str = DEFAULT_JUDGE, effort: str | None = None, max_wait: float = 120.0):
        from openai import AsyncOpenAI, RateLimitError
        llm_factory, self._Faithfulness, self._FactualCorrectness = load_ragas()
        self.model, self.max_wait = model, max_wait
        self.tokens = 0
        self.client = AsyncOpenAI(api_key=config.GROQ_API_KEY, base_url="https://api.groq.com/openai/v1", max_retries=0)
        self._rate_limit_error = RateLimitError
        raw_create = self.client.chat.completions.create

        async def create(*a, **k):                         # pace + retry per-minute limits, surface daily ones
            for attempt in range(6):
                try:
                    r = await raw_create(*a, **k)
                    u = getattr(r, "usage", None)
                    self.tokens += getattr(u, "total_tokens", 0) or 0
                    return r
                except RateLimitError as e:
                    msg = str(e)
                    if "per day" in msg or "(TPD)" in msg or "(RPD)" in msg:
                        raise JudgeBudgetExhausted(msg[:200]) from e
                    await asyncio.sleep(min(_wait_seconds(msg), self.max_wait))
            raise JudgeBudgetExhausted("per-minute rate limit did not clear after 6 retries")

        self.client.chat.completions.create = create
        # ONE event loop for the judge's whole life: the HTTP client is bound to the loop it first runs on, so a fresh
        # asyncio.run() per call fails with "Event loop is closed" from the second call on.
        self._loop = asyncio.new_event_loop()
        kw = dict(max_tokens=MAX_TOKENS, temperature=0)
        if effort:
            kw["reasoning_effort"] = effort
        self.llm = llm_factory(model, provider="openai", client=self.client, **kw)

    # raw RAGAS scores -------------------------------------------------------------------------------------------
    async def _compute(self, tc: EvalCase, needed: set[str]) -> dict:
        got = tc.__dict__.setdefault("_raw", {})          # cached on the case itself (id() can be reused)
        if "faithfulness" in needed and "faithfulness" not in got:
            r = await self._Faithfulness(llm=self.llm).ascore(
                user_input=tc.question, response=tc.response, retrieved_contexts=tc.contexts)
            got["faithfulness"] = float(r.value)
        wanted_fact = {"fact_precision", "fact_recall", "fact_coverage"} & needed
        missing = wanted_fact - got.keys()
        if missing:
            got.update(await self._factual(tc, missing))
        return got

    async def _factual(self, tc: EvalCase, missing: set[str]) -> dict:
        """RAGAS FactualCorrectness building blocks from at most TWO claim passes (2 judge calls each), computed only as needed.

        pass 1: the response's claims checked against the reference   -> precision, and the tp/fp counts
        pass 2: the reference's claims checked against the response   -> coverage (share of reference claims the response states)
        `fact_recall` is RAGAS's own recall, tp/(tp+fn), where tp comes from pass 1 (so it is 0 whenever no RESPONSE claim is
        supported by the reference, even if every reference claim is covered): kept for transparency, not used as completeness.
        Formulas and the 2-decimal rounding are RAGAS's. An empty claim list is a failed measurement, never a score of 0.
        """
        import numpy as np
        fc = self._FactualCorrectness(llm=self.llm, mode="f1")
        out: dict = {}
        resp_vs_ref = ref_vs_resp = None
        if {"fact_precision", "fact_recall"} & missing:
            resp_vs_ref = await fc._decompose_and_verify_claims(tc.response, tc.reference)
            if len(resp_vs_ref) == 0:
                raise ValueError("RAGAS produced no verifiable claims for the response")
        if {"fact_coverage", "fact_recall"} & missing:
            ref_vs_resp = await fc._decompose_and_verify_claims(tc.reference, tc.response)
            if len(ref_vs_resp) == 0:
                raise ValueError("RAGAS produced no verifiable claims for the reference")
        if resp_vs_ref is not None:
            tp, fp = int(np.sum(resp_vs_ref)), int(np.sum(~resp_vs_ref))
            out["fact_precision"] = float(np.round(tp / (tp + fp + 1e-8), 2))
        if ref_vs_resp is not None:
            out["fact_coverage"] = float(np.round(np.mean(ref_vs_resp), 2))
        if resp_vs_ref is not None and ref_vs_resp is not None:
            tp, fn = int(np.sum(resp_vs_ref)), int(np.sum(~ref_vs_resp))
            out["fact_recall"] = float(np.round(tp / (tp + fn + 1e-8), 2))
        return {k: v for k, v in out.items() if k in missing}

    def raw(self, tc: EvalCase, needed: set[str]) -> dict:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return self._loop.run_until_complete(self._compute(tc, needed))
            except Exception as e:
                if is_daily_budget_error(e):
                    raise JudgeBudgetExhausted(str(e)[:200]) from e
                raise
        raise RuntimeError("RagasJudge.raw() cannot be called from a running event loop")

    def close(self) -> None:
        if not self._loop.is_closed():
            self._loop.run_until_complete(self.client.close())
            self._loop.close()

    def metrics(self) -> dict:
        """{name: object with .measure(EvalCase) and .score}, the interface eval/generation.score_case expects."""
        return {name: _Metric(self, name) for name in ("faithfulness", "hallucination", "correctness", "completeness")}


_NEEDS = {"faithfulness": {"faithfulness"}, "hallucination": {"faithfulness"},
          "correctness": {"fact_precision", "fact_coverage"}, "completeness": {"fact_coverage"}}


class _Metric:
    def __init__(self, judge: RagasJudge, name: str):
        self.judge, self.name, self.score = judge, name, None

    def measure(self, tc: EvalCase) -> None:
        got = self.judge.raw(tc, _NEEDS[self.name])
        if self.name == "faithfulness":
            self.score = got["faithfulness"]
        elif self.name == "hallucination":
            self.score = 1.0 - got["faithfulness"]
        elif self.name == "completeness":
            self.score = got["fact_coverage"]
        else:
            self.score = f1(got["fact_precision"], got["fact_coverage"])
