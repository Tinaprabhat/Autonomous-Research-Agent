"""
Unit tests for backend/app/agents/research_agent.py

Only VectorStore.load() is patched (prevents FAISS file I/O).
The retriever, reranker, and LLM are replaced with mocks after construction.
"""
import sys
import pytest
from unittest.mock import MagicMock, patch


FAKE_CHUNKS = [
    {"text": "Self-attention lets the model attend to all positions.", "chunk_id": 0, "source": "paper_a.pdf"},
    {"text": "Transformers replaced recurrent architectures.", "chunk_id": 1, "source": "paper_a.pdf"},
    {"text": "BERT uses masked language modelling.", "chunk_id": 2, "source": "paper_b.pdf"},
]

FAKE_ANSWER = "Attention allows models to focus on relevant tokens."


@pytest.fixture(scope="module")
def agent():
    sys.modules.pop("backend.app.agents.research_agent", None)

    with patch("backend.app.rag_pipeline.vector_store.VectorStore.load", return_value=None), \
         patch("backend.app.rag_pipeline.embedder.Embedder.__init__",    return_value=None), \
         patch("backend.app.rag_pipeline.reranker.Reranker.__init__",    return_value=None), \
         patch("backend.app.llm.local_llm.LocalLLM.__init__",            return_value=None):

        from backend.app.agents.research_agent import ResearchAgent
        ag = ResearchAgent()

    # Wire fakes onto the constructed agent (patching __init__ leaves some
    # sub-objects uninitialised; replace them wholesale)
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = FAKE_CHUNKS

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = FAKE_CHUNKS[:3]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = FAKE_ANSWER

    ag.retriever = mock_retriever
    ag.reranker  = mock_reranker
    ag.llm       = mock_llm

    return ag


# ── run() ─────────────────────────────────────────────────────────────────────

def test_run_returns_string(agent):
    result = agent.run("What is attention?")
    assert isinstance(result, str)


def test_run_returns_nonempty_answer(agent):
    result = agent.run("How does BERT work?")
    assert len(result.strip()) > 0


def test_run_returns_fake_answer(agent):
    result = agent.run("anything")
    assert result == FAKE_ANSWER


def test_run_calls_retriever(agent):
    agent.retriever.search.reset_mock()
    agent.run("What is attention?")
    agent.retriever.search.assert_called_once()


def test_run_calls_reranker(agent):
    agent.reranker.rerank.reset_mock()
    agent.run("What is attention?")
    agent.reranker.rerank.assert_called_once()


def test_run_calls_llm_generate(agent):
    agent.llm.generate.reset_mock()
    agent.run("What is attention?")
    agent.llm.generate.assert_called_once()


def test_run_prompt_contains_query(agent):
    agent.llm.generate.reset_mock()
    query = "unique_query_string_12345"
    agent.run(query)
    prompt = agent.llm.generate.call_args[0][0]
    assert query in prompt


def test_run_prompt_contains_context(agent):
    agent.llm.generate.reset_mock()
    agent.run("What is attention?")
    prompt = agent.llm.generate.call_args[0][0]
    assert any(chunk["text"][:20] in prompt for chunk in FAKE_CHUNKS[:3])


def test_run_retrieves_with_k_20(agent):
    agent.retriever.search.reset_mock()
    agent.run("test query")
    call = agent.retriever.search.call_args
    k_used = call[0][1] if len(call[0]) > 1 else call[1].get("k")
    assert k_used == 20


def test_run_reranks_to_top_5(agent):
    agent.reranker.rerank.reset_mock()
    agent.run("test query")
    call = agent.reranker.rerank.call_args
    top_k = call[1].get("top_k") if call[1] else call[0][2]
    assert top_k == 5
