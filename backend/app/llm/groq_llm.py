"""
groq_llm.py — Groq-hosted LLM (primary generator).

Same interface as OllamaLLM: `.generate(prompt)` and `.generate_stream(prompt)`.
"""
from __future__ import annotations

from backend.app import config


class LLMError(RuntimeError):
    """Any generation failure (API error, timeout, empty response)."""


class LLMRateLimitError(LLMError):
    """HTTP 429. `daily` is True when a per-DAY quota (tokens/requests) is exhausted, not a per-minute one."""

    def __init__(self, message: str, retry_after: float | None = None, daily: bool = False):
        super().__init__(message)
        self.retry_after = retry_after
        self.daily = daily


def _classify_429(e: Exception) -> "LLMRateLimitError | None":
    """Turn a Groq 429 into LLMRateLimitError (parses 'try again in 9m47s' and TPD/RPD)."""
    import re
    msg = str(e)
    if "429" not in msg and type(e).__name__ != "RateLimitError":
        return None
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", msg)
    wait = None
    if m:
        wait = float(m.group(3)) + 60 * int(m.group(2) or 0) + 3600 * int(m.group(1) or 0)
    daily = bool(re.search(r"\((TPD|RPD)\)|per day", msg))
    return LLMRateLimitError(f"Groq rate limit: {msg[:300]}", retry_after=wait, daily=daily)


_UNICODE_FIXES = {"‑": "-", "‐": "-", " ": " ", " ": " "}


def _normalise(text: str, strip: bool = True) -> str:
    for k, v in _UNICODE_FIXES.items():
        text = text.replace(k, v)
    return text.strip() if strip else text


class GroqLLM:

    def __init__(self, model: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 400, timeout: float = 30.0,
                 api_key: str | None = None, client=None):
        self.model = model or config.GROQ_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = client
        self._api_key = api_key or config.GROQ_API_KEY

    @property
    def available(self) -> bool:
        return bool(self._client or self._api_key)

    def _get_client(self):
        if self._client is None:
            if not self._api_key:
                raise LLMError("GROQ_API_KEY is not set")
            from groq import Groq
            # max_retries=0: retry policy is owned by the pipeline, not hidden in the SDK
            self._client = Groq(api_key=self._api_key, timeout=self.timeout, max_retries=0)
        return self._client

    def _kwargs(self, prompt: str, stream: bool, max_tokens: int | None = None) -> dict:
        extra = {"reasoning_effort": "low"} if "gpt-oss" in self.model else {}
        return dict(
            **extra,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=stream,
        )

    def generate(self, prompt: str, use_cache: bool = True, max_tokens: int | None = None) -> str:
        """`max_tokens` overrides the default output budget for this call (it also covers hidden reasoning tokens)."""
        try:
            resp = self._get_client().chat.completions.create(**self._kwargs(prompt, False, max_tokens))
            text = _normalise(resp.choices[0].message.content or "")
        except LLMError:
            raise
        except Exception as e:
            raise (_classify_429(e) or LLMError(f"Groq error: {type(e).__name__}: {e}")) from e
        if not text:
            raise LLMError("Groq returned an empty response")
        return text

    def generate_stream(self, prompt: str):
        try:
            stream = self._get_client().chat.completions.create(**self._kwargs(prompt, True))
            got = False
            for part in stream:
                tok = part.choices[0].delta.content if part.choices else None
                if tok:
                    got = True
                    yield _normalise(tok, strip=False)
        except LLMError:
            raise
        except Exception as e:
            raise (_classify_429(e) or LLMError(f"Groq stream error: {type(e).__name__}: {e}")) from e
        if not got:
            raise LLMError("Groq stream produced no tokens")
