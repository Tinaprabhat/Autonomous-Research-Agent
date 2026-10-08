"""
judge.py — DeepEval LLM-as-judge backed by Groq (no OpenAI key needed).

Known tradeoff (see docs/known_limits.md): the judge is from the same provider/family
used for generation, so mild self-preference bias is possible. Acceptable at this scale.
"""
from __future__ import annotations

import json
import re
import time

from deepeval.models import DeepEvalBaseLLM

from backend.app import config

JUDGE_MODEL = config._env("JUDGE_MODEL", default="openai/gpt-oss-20b")   # own daily quota, separate from the generator


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


class GroqJudge(DeepEvalBaseLLM):

    def __init__(self, model: str | None = None, client=None, max_retries: int = 8,
                 min_interval: float = 1.0):
        self._model_name = model or JUDGE_MODEL
        self._client = client
        self.max_retries = max_retries
        self.min_interval = min_interval
        super().__init__(self._model_name)

    def load_model(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=config.GROQ_API_KEY, max_retries=0, timeout=60)
        return self._client

    def get_model_name(self) -> str:
        return f"groq/{self._model_name}"

    def _call(self, prompt: str, json_mode: bool) -> str:
        """One judge call with rate-limit aware retries (Groq free tier: RPM and TPM caps)."""
        kw = {"reasoning_effort": "low"} if "gpt-oss" in self._model_name else {}
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        delay = 3.0
        for attempt in range(self.max_retries):
            try:
                r = self.model.chat.completions.create(
                    model=self._model_name, temperature=0,
                    messages=[{"role": "user", "content": prompt}], max_tokens=2000, **kw)
                time.sleep(self.min_interval)
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                from backend.app.llm.groq_llm import _classify_429
                rl = _classify_429(e)
                if rl is not None and rl.daily and (rl.retry_after is None or rl.retry_after > 180):
                    from eval.generation import JudgeBudgetExhausted
                    raise JudgeBudgetExhausted(str(rl)[:200]) from e
                if attempt == self.max_retries - 1:
                    raise
                msg = str(e)
                m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", msg)
                if m:     # server told us exactly how long to wait
                    wait = float(m.group(2)) + 60 * int(m.group(1) or 0) + 1.0
                else:
                    wait = delay
                    delay = min(delay * 2, 60)
                time.sleep(min(wait, 120))
        raise RuntimeError("unreachable")

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return self._call(prompt, json_mode=False)
        text = self._call(prompt + "\n\nRespond with a single valid JSON object only.", json_mode=True)
        try:
            return schema.model_validate(_extract_json(text))
        except Exception:
            fixed = self._call(
                "Fix this so it is a single valid JSON object matching the requested schema. "
                "Return JSON only.\n\n" + text, json_mode=True)
            return schema.model_validate(_extract_json(fixed))

    async def a_generate(self, prompt: str, schema=None):
        import asyncio
        return await asyncio.to_thread(self.generate, prompt, schema)
