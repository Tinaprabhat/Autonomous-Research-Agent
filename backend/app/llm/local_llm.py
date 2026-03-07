import requests
import hashlib
import time
import json


class LocalLLM:

    # Module-level cache shared across all instances
    _cache: dict = {}
    CACHE_TTL = 3600  # cached responses expire after 1 hour

    def __init__(self, model="mistral", temperature=0.3, max_tokens=512, timeout=60):
        self.model = model
        self.url = "http://localhost:11434/api/generate"
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

    def generate(self, prompt: str, use_cache: bool = True) -> str:
        """
        Returns full response as a string.
        Caches deterministic calls (temperature=0) automatically.
        keep_alive=10m tells Ollama to keep the model in RAM between calls,
        saving ~2s reload time on every request.
        """
        key = self._cache_key(prompt)
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
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "keep_alive": "10m",
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
            raise RuntimeError("Ollama is not running. Start it with: ollama serve")
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama timed out after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}")

    # ── Generate (streaming) ──────────────────────────────────────────────────

    def generate_stream(self, prompt: str):
        """
        Yields tokens as they are produced — first token appears in ~1-2s
        instead of waiting 30s for the full response.

        Use in agent/UI layer:
            for token in llm.generate_stream(prompt):
                print(token, end="", flush=True)
        """
        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "keep_alive": "10m",
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
            raise RuntimeError("Ollama is not running. Start it with: ollama serve")
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama timed out after {self.timeout}s")

    # ── Cache utils ───────────────────────────────────────────────────────────

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()

    @classmethod
    def cache_stats(cls) -> dict:
        now = time.time()
        valid = sum(1 for _, ts in cls._cache.values() if now - ts < cls.CACHE_TTL)
        return {"total": len(cls._cache), "valid": valid, "expired": len(cls._cache) - valid}