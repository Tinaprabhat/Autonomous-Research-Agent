"""
router.py — GroqLLM primary, OllamaLLM fallback, one shared interface.

Provider-level fallback lives here (Groq fails -> try Ollama). The call-level
"retry once, then warn" policy lives in the pipeline, which calls this router.
"""
from __future__ import annotations

import logging
import time

from backend.app import config
from backend.app.llm.groq_llm import GroqLLM, LLMError, LLMRateLimitError
from backend.app.llm.local_llm import OllamaLLM

log = logging.getLogger(__name__)


class LLMRouter:

    def __init__(self, primary=None, fallback=None, extra=None):
        """primary -> extra (more Groq models, own quotas) -> fallback (Ollama)."""
        self.primary = primary if primary is not None else GroqLLM()
        if extra is None:
            extra = [GroqLLM(model=m) for m in config.GROQ_FALLBACK_MODELS] if primary is None else []
        self.extra = list(extra)
        self.fallback = fallback if fallback is not None else OllamaLLM(
            temperature=0.2, max_tokens=300, timeout=60)
        self.last_provider: str | None = None
        self._cooldown: dict[str, float] = {}     # provider name -> monotonic time until which it is skipped

    @property
    def model(self) -> str:
        return getattr(self.primary, "model", "unknown")

    def _providers(self):
        """Available providers in order, skipping any that are cooling down after a 429
        (unless every provider is cooling down, in which case try them all rather than fail fast)."""
        chain = []
        if getattr(self.primary, "available", True):
            chain.append(("groq", self.primary))
        for llm in self.extra:
            if getattr(llm, "available", True):
                chain.append((f"groq:{llm.model}", llm))
        if self.fallback is not None:
            chain.append(("ollama", self.fallback))
        now = time.monotonic()
        live = [(n, p) for n, p in chain if self._cooldown.get(n, 0) <= now]
        yield from (live or chain)

    def _note_failure(self, name: str, e: Exception) -> None:
        if isinstance(e, LLMRateLimitError):
            wait = e.retry_after if e.retry_after else 60
            self._cooldown[name] = time.monotonic() + min(max(wait, 30), 900)

    def generate(self, prompt: str, use_cache: bool = True, max_tokens: int | None = None) -> str:
        errors = []
        for name, llm in self._providers():
            try:
                out = llm.generate(prompt) if max_tokens is None else llm.generate(prompt, max_tokens=max_tokens)
                if not out or not out.strip():
                    raise LLMError(f"{name} returned an empty response")
                self.last_provider = name
                return out
            except Exception as e:
                log.warning("LLM provider %s failed: %s", name, e)
                self._note_failure(name, e)
                errors.append(f"{name}: {e}")
        raise LLMError("; ".join(errors) or "no LLM provider configured")

    def generate_stream(self, prompt: str):
        """Falls back only if a provider fails before emitting any token."""
        errors = []
        for name, llm in self._providers():
            started = False
            try:
                for tok in llm.generate_stream(prompt):
                    started = True
                    self.last_provider = name
                    yield tok
                return
            except Exception as e:
                if started:
                    raise LLMError(f"{name} stream interrupted: {e}") from e
                log.warning("LLM provider %s failed to stream: %s", name, e)
                self._note_failure(name, e)
                errors.append(f"{name}: {e}")
        raise LLMError("; ".join(errors) or "no LLM provider configured")


def cheap_rewrite_llm() -> GroqLLM:
    """Small/fast Groq model for query rewriting."""
    return GroqLLM(model=config.GROQ_REWRITE_MODEL, temperature=0.0,
                   max_tokens=300, timeout=10.0)
