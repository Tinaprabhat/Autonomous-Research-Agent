from backend.app.citations.formatter import (
    arxiv_pdf_url, arxiv_url, format_context_with_citations, is_arxiv_id)


def test_is_arxiv_id():
    assert is_arxiv_id("2401.12345") and is_arxiv_id("1901.09890") and is_arxiv_id("cs.LG/0701001")
    assert not is_arxiv_id("some_local_file") and not is_arxiv_id(None) and not is_arxiv_id("")


def test_arxiv_url_from_provenance():
    assert arxiv_url("2401.12345") == "https://arxiv.org/abs/2401.12345"
    assert arxiv_url("local-paper") is None


def test_pdf_url_with_version_and_page():
    assert arxiv_pdf_url("2401.12345", "v2", 7) == "https://arxiv.org/pdf/2401.12345v2#page=7"
    assert arxiv_pdf_url("2401.12345") == "https://arxiv.org/pdf/2401.12345"


def test_context_is_numbered_and_labelled():
    chunks = [{"text": "alpha", "title": "T1", "arxiv_id": "2401.12345", "section": "Method", "page": 3},
              {"text": "beta", "source": "x/y/legacy.pdf"}]
    ctx, numbered = format_context_with_citations(chunks)
    assert ctx.startswith("[1] (T1 — arXiv:2401.12345, Method, p.3)\nalpha")
    assert "[2] (legacy)\nbeta" in ctx
    assert numbered is chunks
