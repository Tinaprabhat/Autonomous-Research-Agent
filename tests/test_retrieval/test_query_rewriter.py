from backend.app.llm.groq_llm import LLMError
from backend.app.retrieval.query_rewriter import QueryRewriter


class LLM:
    def __init__(self, out=None, exc=None):
        self.out, self.exc = out, exc
    def generate(self, prompt):
        if self.exc:
            raise self.exc
        return self.out


def test_rewrite_expands_query_and_keeps_original_terms():
    out = QueryRewriter(LLM("retrieval-augmented generation RAG LLM grounding")).rewrite("what is RAG?")
    assert "what is RAG?" in out and "grounding" in out


def test_llm_failure_returns_original():
    assert QueryRewriter(LLM(exc=LLMError("boom"))).rewrite("q?") == "q?"


def test_empty_output_returns_original():
    assert QueryRewriter(LLM("")).rewrite("q?") == "q?"


def test_runaway_output_returns_original():
    assert QueryRewriter(LLM("word " * 200)).rewrite("q?") == "q?"


def test_none_llm_is_noop():
    assert QueryRewriter(None).rewrite("q?") == "q?"


def test_takes_first_line_only():
    out = QueryRewriter(LLM('"meta learning MAML"\nExplanation: blah')).rewrite("x")
    assert "Explanation" not in out
