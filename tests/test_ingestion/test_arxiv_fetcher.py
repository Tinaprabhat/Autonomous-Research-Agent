"""
Unit tests for backend/app/ingestion/arxiv_fetcher.py

Heavy network calls (arxiv.Search, result.download_pdf) are mocked so the
suite runs offline and deterministically.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from backend.app.ingestion.arxiv_fetcher import search_papers, download_papers


# ── Helpers / fixtures ────────────────────────────────────────────────────────

def _make_fake_result(title="Attention Is All You Need", pdf_url="https://arxiv.org/pdf/1706.03762"):
    r = MagicMock()
    r.title = title
    r.authors = [MagicMock(name="A. Author"), MagicMock(name="B. Author")]
    r.summary = "A transformer paper."
    r.pdf_url = pdf_url
    r.published.strftime.return_value = "2017-06-12"
    return r


# ── search_papers ─────────────────────────────────────────────────────────────

@patch("backend.app.ingestion.arxiv_fetcher.arxiv.Search")
def test_search_papers_returns_list(mock_search):
    fake = _make_fake_result()
    mock_search.return_value.results.return_value = [fake]

    results = search_papers("attention mechanism", max_results=1)
    assert isinstance(results, list)
    assert len(results) == 1


@patch("backend.app.ingestion.arxiv_fetcher.arxiv.Search")
def test_search_papers_result_has_required_keys(mock_search):
    fake = _make_fake_result()
    mock_search.return_value.results.return_value = [fake]

    results = search_papers("transformers")
    paper = results[0]
    for key in ("title", "authors", "summary", "pdf_url", "published"):
        assert key in paper, f"Missing key: {key}"


@patch("backend.app.ingestion.arxiv_fetcher.arxiv.Search")
def test_search_papers_empty_results(mock_search):
    mock_search.return_value.results.return_value = []
    assert search_papers("xyzzy obscure query 99999") == []


@patch("backend.app.ingestion.arxiv_fetcher.arxiv.Search")
def test_search_papers_max_results_respected(mock_search):
    fakes = [_make_fake_result(title=f"Paper {i}") for i in range(10)]
    mock_search.return_value.results.return_value = fakes[:3]

    results = search_papers("anything", max_results=3)
    assert len(results) <= 3


@patch("backend.app.ingestion.arxiv_fetcher.arxiv.Search")
def test_search_papers_authors_are_list(mock_search):
    fake = _make_fake_result()
    mock_search.return_value.results.return_value = [fake]

    results = search_papers("test")
    assert isinstance(results[0]["authors"], list)


# ── download_papers ───────────────────────────────────────────────────────────

@patch("backend.app.ingestion.arxiv_fetcher.arxiv.Client")
@patch("backend.app.ingestion.arxiv_fetcher.arxiv.Search")
@patch("os.path.exists", return_value=False)
def test_download_papers_calls_download(mock_exists, mock_search, mock_client, tmp_path):
    fake_result = MagicMock()
    mock_client.return_value.results.return_value = iter([fake_result])
    mock_search.return_value = MagicMock()

    papers = [{"pdf_url": "https://arxiv.org/pdf/1706.03762", "title": "Test Paper"}]

    with patch("backend.app.ingestion.arxiv_fetcher.DOWNLOAD_DIR", str(tmp_path)):
        download_papers(papers)

    fake_result.download_pdf.assert_called_once()


@patch("os.path.exists", return_value=True)
def test_download_papers_skips_existing_files(mock_exists, tmp_path):
    papers = [{"pdf_url": "https://arxiv.org/pdf/1706.03762", "title": "Already Downloaded"}]
    with patch("backend.app.ingestion.arxiv_fetcher.DOWNLOAD_DIR", str(tmp_path)):
        with patch("backend.app.ingestion.arxiv_fetcher.arxiv.Client") as mock_client:
            download_papers(papers)
            mock_client.assert_not_called()


@patch("backend.app.ingestion.arxiv_fetcher.arxiv.Client")
@patch("backend.app.ingestion.arxiv_fetcher.arxiv.Search")
@patch("os.path.exists", return_value=False)
def test_download_papers_adds_file_path(mock_exists, mock_search, mock_client, tmp_path):
    fake_result = MagicMock()
    mock_client.return_value.results.return_value = iter([fake_result])
    mock_search.return_value = MagicMock()

    papers = [{"pdf_url": "https://arxiv.org/pdf/1706.03762", "title": "Test Paper"}]
    with patch("backend.app.ingestion.arxiv_fetcher.DOWNLOAD_DIR", str(tmp_path)):
        result = download_papers(papers)

    assert "file_path" in result[0]
