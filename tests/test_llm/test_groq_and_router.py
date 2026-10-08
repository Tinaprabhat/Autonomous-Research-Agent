"""Tests for groq_llm.py and router.py (fake clients; no network)."""
from types import SimpleNamespace

import pytest

from backend.app.llm.groq_llm import GroqLLM, LLMError
from backend.app.llm.router import LLMRouter


def _resp(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


# ── GroqLLM ──────────────────────────────────────────────────────────────────

def test_generate_returns_stripped_text():
    llm = GroqLLM(client=_client(lambda **kw: _resp("  hello  ")))
    assert llm.generate("p") == "hello"


def test_generate_sends_model_and_prompt():
    seen = {}
    def create(**kw):
        seen.update(kw)
        return _resp("ok")
    GroqLLM(model="m1", client=_client(create)).generate("the prompt")
    assert seen["model"] == "m1" and seen["messages"][0]["content"] == "the prompt"
    assert seen["stream"] is False


def test_api_error_becomes_llm_error():
    def create(**kw):
        raise TimeoutError("slow")
    with pytest.raises(LLMError, match="Groq error"):
        GroqLLM(client=_client(create)).generate("p")


def test_empty_response_is_llm_error():
    with pytest.raises(LLMError, match="empty"):
        GroqLLM(client=_client(lambda **kw: _resp("   "))).generate("p")


def test_missing_key_is_llm_error():
    llm = GroqLLM(api_key=None)
    llm._api_key = None
    assert not llm.available
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        llm.generate("p")


def test_stream_yields_tokens():
    def create(**kw):
        assert kw["stream"] is True
        return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))])
                     for t in ["Hel", "lo", None, "!"]])
    assert "".join(GroqLLM(client=_client(create)).generate_stream("p")) == "Hello!"


def test_stream_with_no_tokens_errors():
    with pytest.raises(LLMError):
        list(GroqLLM(client=_client(lambda **kw: iter([]))).generate_stream("p"))


# ── Router ───────────────────────────────────────────────────────────────────

class Fake:
    available = True
    def __init__(self, out=None, exc=None, toks=None, model="fake"):
        self.out, self.exc, self.toks, self.model, self.calls = out, exc, toks, model, 0
    def generate(self, prompt):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.out
    def generate_stream(self, prompt):
        self.calls += 1
        if self.exc:
            raise self.exc
        yield from self.toks


def test_router_uses_primary_when_healthy():
    p, f = Fake("from groq"), Fake("from ollama")
    r = LLMRouter(p, f)
    assert r.generate("q") == "from groq" and r.last_provider == "groq" and f.calls == 0


def test_router_falls_back_to_ollama():
    r = LLMRouter(Fake(exc=LLMError("down")), Fake("from ollama"))
    assert r.generate("q") == "from ollama" and r.last_provider == "ollama"


def test_router_skips_unavailable_primary():
    p = Fake("x"); p.available = False
    r = LLMRouter(p, Fake("local"))
    assert r.generate("q") == "local" and p.calls == 0


def test_router_raises_when_all_fail():
    r = LLMRouter(Fake(exc=LLMError("a")), Fake(exc=LLMError("b")))
    with pytest.raises(LLMError, match="groq: a; ollama: b"):
        r.generate("q")


def test_router_empty_primary_output_falls_back():
    assert LLMRouter(Fake("  "), Fake("real")).generate("q") == "real"


def test_stream_falls_back_before_first_token():
    r = LLMRouter(Fake(exc=LLMError("down")), Fake(toks=["a", "b"]))
    assert "".join(r.generate_stream("q")) == "ab"


def test_stream_does_not_fall_back_after_tokens_emitted():
    class Broken(Fake):
        def generate_stream(self, prompt):
            yield "partial"
            raise LLMError("cut")
    r = LLMRouter(Broken(), Fake(toks=["x"]))
    got = []
    with pytest.raises(LLMError, match="interrupted"):
        for t in r.generate_stream("q"):
            got.append(t)
    assert got == ["partial"]


# ── 429 handling + multi-model chain ─────────────────────────────────────────

def test_groq_429_becomes_rate_limit_error_with_daily_flag():
    from backend.app.llm.groq_llm import LLMRateLimitError

    class RateLimitError(Exception):
        pass

    def create(**kw):
        raise RateLimitError("Error code: 429 - Rate limit reached on tokens per day (TPD): Limit 200000, "
                             "Used 199640. Please try again in 9m47.088s.")
    with pytest.raises(LLMRateLimitError) as ei:
        GroqLLM(client=_client(create)).generate("p")
    assert ei.value.daily and ei.value.retry_after == pytest.approx(587.088)


def test_per_minute_429_is_not_daily():
    from backend.app.llm.groq_llm import LLMRateLimitError

    class RateLimitError(Exception):
        pass

    def create(**kw):
        raise RateLimitError("Error code: 429 - tokens per minute (TPM) ... try again in 3.2s")
    with pytest.raises(LLMRateLimitError) as ei:
        GroqLLM(client=_client(create)).generate("p")
    assert not ei.value.daily and ei.value.retry_after == pytest.approx(3.2)


def test_router_walks_groq_models_before_ollama():
    from backend.app.llm.groq_llm import LLMRateLimitError
    p = Fake(exc=LLMRateLimitError("TPD", daily=True))
    x1 = Fake(exc=LLMRateLimitError("TPD", daily=True)); x1.model = "m1"
    x2 = Fake("from third groq model"); x2.model = "m2"
    o = Fake("ollama")
    r = LLMRouter(p, o, extra=[x1, x2])
    assert r.generate("q") == "from third groq model" and r.last_provider == "groq:m2" and o.calls == 0


def test_router_default_chain_comes_from_config(monkeypatch):
    monkeypatch.setattr("backend.app.config.GROQ_FALLBACK_MODELS", ["a/b", "c/d"])
    r = LLMRouter(fallback=Fake("x"))
    assert [e.model for e in r.extra] == ["a/b", "c/d"]


def test_rate_limited_provider_is_skipped_during_cooldown():
    from backend.app.llm.groq_llm import LLMRateLimitError
    p = Fake(exc=LLMRateLimitError("TPD", retry_after=600, daily=True))
    x = Fake("from second"); x.model = "m2"
    r = LLMRouter(p, Fake("ollama"), extra=[x])
    assert r.generate("q") == "from second" and p.calls == 1
    assert r.generate("q") == "from second" and p.calls == 1       # not retried: still cooling down


def test_cooldown_does_not_apply_to_non_rate_limit_errors():
    p = Fake(exc=LLMError("transient network"))
    r = LLMRouter(p, Fake("ollama"), extra=[])
    r.generate("q"); r.generate("q")
    assert p.calls == 2


def test_if_everything_is_cooling_down_still_tries_all():
    from backend.app.llm.groq_llm import LLMRateLimitError
    p = Fake(exc=LLMRateLimitError("x", retry_after=600, daily=True))
    o = Fake(exc=LLMRateLimitError("y", retry_after=600, daily=True))
    r = LLMRouter(p, o, extra=[])
    with pytest.raises(LLMError):
        r.generate("q")
    o.exc = None; o.out = "recovered"
    assert r.generate("q") == "recovered"              # all cooling -> tried anyway rather than failing fast
