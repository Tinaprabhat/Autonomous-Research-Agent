import requests
import hashlib
import time
import json

from backend.app import config
from backend.app.llm.groq_llm import LLMError


class OllamaLLM:
    """Local Ollama backend (fallback generator)."""

    # Module-level cache shared across all instances
    _cache: dict = {}
    CACHE_TTL = 3600  # cached responses expire after 1 hour

    def __init__(self, model=None, temperature=0.3, max_tokens=512, timeout=60, url=None):
        self.model = model or config.OLLAMA_MODEL
        self.url = url or "http://localhost:11434/api/generate"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_key(self, prompt: str) -> str:
        raw = f"{self.model}:{self.max_tokens}:{prompt}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_cached(self, key: str):
        if key in self._cache:
            response, ts = self._cache[key]
            if time.time() - ts < self.CACHE_TTL:
                return response
            del self._cache[key]
        return None

    def _set_cache(self, key: str, response: str):
        self._cache[key] = (response, time.time())

    # ── Generate (blocking) ───────────────────────────────────────────────────

    def generate(self, prompt: str, use_cache: bool = True, max_tokens: int | None = None) -> str:
        """
        Returns full response as a string.
        Caches deterministic calls (temperature=0) automatically.
        keep_alive tells Ollama to keep the model in RAM between calls.
        """
        budget = max_tokens or self.max_tokens
        key = self._cache_key(prompt) + f":{budget}"
        if use_cache and self.temperature == 0.0:
            cached = self._get_cached(key)
            if cached is not None:
                return cached

        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": budget,
                    }
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            if "response" not in data:
                raise ValueError(f"Unexpected Ollama response: {data}")

            result = data["response"].strip()

            if use_cache:
                self._set_cache(key, result)

            return result

        except requests.exceptions.ConnectionError:
            raise LLMError("Ollama is not running. Start it with: ollama serve")
        except requests.exceptions.Timeout:
            raise LLMError(f"Ollama timed out after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            raise LLMError(f"Ollama HTTP error: {e}")

    # ── Generate (streaming) ──────────────────────────────────────────────────

    def generate_stream(self, prompt: str):
        """Yields tokens as they are produced."""
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "keep_alive": "10m",
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    }
                },
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break

        except requests.exceptions.ConnectionError:
            raise LLMError("Ollama is not running. Start it with: ollama serve")
        except requests.exceptions.Timeout:
            raise LLMError(f"Ollama timed out after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            raise LLMError(f"Ollama HTTP error: {e}")

    # [DEAD CODE] cache utilities (checked 2026-10-15): no production code called them; only their own tests did.
    # ── Cache utils ───────────────────────────────────────────────────────────
    #
    # @classmethod
    # def clear_cache(cls):
    #     cls._cache.clear()
    #
    # @classmethod
    # def cache_stats(cls) -> dict:
    #     now = time.time()
    #     valid = sum(1 for _, ts in cls._cache.values() if now - ts < cls.CACHE_TTL)
    #     return {"total": len(cls._cache), "valid": valid, "expired": len(cls._cache) - valid}


# Backwards-compatible name used by older scripts/tests.
LocalLLM = OllamaLLM
