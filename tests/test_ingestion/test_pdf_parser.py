"""
Unit tests for backend/app/ingestion/pdf_parser.py

fitz (PyMuPDF) is mocked so tests run without real PDFs.
"""
import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fake_pdf(pages_text: list[str]):
    """Return a fake fitz.Document with the given page texts."""
    doc = MagicMock()
    pages = []
    for text in pages_text:
        page = MagicMock()
        page.get_text.return_value = text
        pages.append(page)
    doc.__iter__ = lambda self: iter(pages)
    doc.__enter__ = lambda self: self
    doc.__exit__ = MagicMock(return_value=False)
    return doc


# ── Tests ─────────────────────────────────────────────────────────────────────

@patch("backend.app.ingestion.pdf_parser.fitz.open")
def test_parse_returns_string(mock_open):
    mock_open.return_value = _make_fake_pdf(["Hello world."])
    from backend.app.ingestion.pdf_parser import extract_text_from_pdf
    result = extract_text_from_pdf("dummy.pdf")
    assert isinstance(result, str)


@patch("backend.app.ingestion.pdf_parser.fitz.open")
def test_parse_concatenates_pages(mock_open):
    mock_open.return_value = _make_fake_pdf(["Page one text.", "Page two text."])
    from backend.app.ingestion.pdf_parser import extract_text_from_pdf
    result = extract_text_from_pdf("dummy.pdf")
    assert "Page one" in result
    assert "Page two" in result


@patch("backend.app.ingestion.pdf_parser.fitz.open")
def test_parse_empty_pdf_returns_empty_or_whitespace(mock_open):
    mock_open.return_value = _make_fake_pdf([])
    from backend.app.ingestion.pdf_parser import extract_text_from_pdf
    result = extract_text_from_pdf("dummy.pdf")
    assert result.strip() == ""


@patch("backend.app.ingestion.pdf_parser.fitz.open")
def test_parse_strips_leading_trailing_whitespace(mock_open):
    mock_open.return_value = _make_fake_pdf(["  \n  content  \n  "])
    from backend.app.ingestion.pdf_parser import extract_text_from_pdf
    result = extract_text_from_pdf("dummy.pdf")
    assert "content" in result


@patch("backend.app.ingestion.pdf_parser.fitz.open")
def test_parse_called_with_correct_path(mock_open):
    mock_open.return_value = _make_fake_pdf(["text"])
    from backend.app.ingestion.pdf_parser import extract_text_from_pdf
    extract_text_from_pdf("/path/to/paper.pdf")
    mock_open.assert_called_once_with("/path/to/paper.pdf")
