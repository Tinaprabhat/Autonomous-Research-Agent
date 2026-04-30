"""
Unit tests for backend/app/llm/local_llm.py

All HTTP calls are mocked — no Ollama instance required.
"""
import time
import hashlib
import pytest
from unittest.mock import MagicMock, patch

from backend.app.llm.local_llm import LocalLLM


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_response(text: str, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"response": text}
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture(autouse=True)
def clear_llm_cache():
    LocalLLM.clear_cache()
    yield
    LocalLLM.clear_cache()


@pytest.fixture
def llm():
    return LocalLLM(model="mistral", temperature=0.3, max_tokens=64, timeout=5)


@pytest.fixture
def llm_zero_temp():
    return LocalLLM(model="mistral", temperature=0.0, max_tokens=64, timeout=5)


# ── generate — happy path ─────────────────────────────────────────────────────

@patch("backend.app.llm.local_llm.requests.post")
def test_generate_returns_string(mock_post, llm):
    mock_post.return_value = _mock_response("This is an answer.")
    result = llm.generate("What is attention?")
    assert isinstance(result, str)
    assert result == "This is an answer."


@patch("backend.app.llm.local_llm.requests.post")
def test_generate_strips_whitespace(mock_post, llm):
    mock_post.return_value = _mock_response("  answer with spaces  ")
    result = llm.generate("question")
    assert result == "answer with spaces"


@patch("backend.app.llm.local_llm.requests.post")
def test_generate_sends_correct_model(mock_post, llm):
    mock_post.return_value = _mock_response("ok")
    llm.generate("test")
    payload = mock_post.call_args[1]["json"]
    assert payload["model"] == "mistral"


@patch("backend.app.llm.local_llm.requests.post")
def test_generate_sends_correct_options(mock_post, llm):
    mock_post.return_value = _mock_response("ok")
    llm.generate("test")
    options = mock_post.call_args[1]["json"]["options"]
    assert options["temperature"] == llm.temperature
    assert options["num_predict"] == llm.max_tokens


# ── Caching ───────────────────────────────────────────────────────────────────

@patch("backend.app.llm.local_llm.requests.post")
def test_cache_hit_avoids_second_http_call(mock_post, llm_zero_temp):
    mock_post.return_value = _mock_response("cached answer")
    llm_zero_temp.generate("same prompt", use_cache=True)
    llm_zero_temp.generate("same prompt", use_cache=True)
    assert mock_post.call_count == 1


@patch("backend.app.llm.local_llm.requests.post")
def test_cache_not_used_for_nonzero_temperature(mock_post, llm):
    mock_post.return_value = _mock_response("answer")
    llm.generate("prompt", use_cache=True)
    llm.generate("prompt", use_cache=True)
    assert mock_post.call_count == 2


@patch("backend.app.llm.local_llm.requests.post")
def test_cache_disabled_makes_two_calls(mock_post, llm_zero_temp):
    mock_post.return_value = _mock_response("answer")
    llm_zero_temp.generate("prompt", use_cache=False)
    llm_zero_temp.generate("prompt", use_cache=False)
    assert mock_post.call_count == 2


@patch("backend.app.llm.local_llm.requests.post")
def test_cache_key_differs_for_different_prompts(mock_post, llm_zero_temp):
    mock_post.return_value = _mock_response("answer")
    llm_zero_temp.generate("prompt A", use_cache=True)
    llm_zero_temp.generate("prompt B", use_cache=True)
    assert mock_post.call_count == 2


def test_cache_stats_structure(llm_zero_temp):
    stats = LocalLLM.cache_stats()
    assert "total" in stats
    assert "valid" in stats
    assert "expired" in stats


@patch("backend.app.llm.local_llm.requests.post")
def test_cache_stats_counts_entries(mock_post, llm_zero_temp):
    mock_post.return_value = _mock_response("answer")
    llm_zero_temp.generate("prompt A", use_cache=True)
    llm_zero_temp.generate("prompt B", use_cache=True)
    stats = LocalLLM.cache_stats()
    assert stats["total"] == 2
    assert stats["valid"] == 2


# ── Error handling ────────────────────────────────────────────────────────────

@patch("backend.app.llm.local_llm.requests.post", side_effect=Exception("Connection refused"))
def test_generate_raises_on_connection_error(mock_post, llm):
    import requests
    mock_post.side_effect = requests.exceptions.ConnectionError()
    with pytest.raises(RuntimeError, match="Ollama is not running"):
        llm.generate("test")


@patch("backend.app.llm.local_llm.requests.post")
def test_generate_raises_on_timeout(mock_post, llm):
    import requests
    mock_post.side_effect = requests.exceptions.Timeout()
    with pytest.raises(RuntimeError, match="timed out"):
        llm.generate("test")


@patch("backend.app.llm.local_llm.requests.post")
def test_generate_raises_on_missing_response_key(mock_post, llm):
    resp = MagicMock()
    resp.json.return_value = {"error": "model not found"}
    resp.raise_for_status = MagicMock()
    mock_post.return_value = resp
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        llm.generate("test")


# ── generate_stream ───────────────────────────────────────────────────────────

@patch("backend.app.llm.local_llm.requests.post")
def test_generate_stream_yields_tokens(mock_post, llm):
    import json
    lines = [
        json.dumps({"response": "Hello", "done": False}).encode(),
        json.dumps({"response": " world", "done": False}).encode(),
        json.dumps({"response": "!", "done": True}).encode(),
    ]
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = lines
    mock_post.return_value = resp

    tokens = list(llm.generate_stream("tell me something"))
    assert tokens == ["Hello", " world", "!"]


@patch("backend.app.llm.local_llm.requests.post")
def test_generate_stream_stops_at_done(mock_post, llm):
    import json
    lines = [
        json.dumps({"response": "A", "done": False}).encode(),
        json.dumps({"response": "B", "done": True}).encode(),
        json.dumps({"response": "C", "done": False}).encode(),  # should not appear
    ]
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.iter_lines.return_value = lines
    mock_post.return_value = resp

    tokens = list(llm.generate_stream("test"))
    assert "C" not in tokens
