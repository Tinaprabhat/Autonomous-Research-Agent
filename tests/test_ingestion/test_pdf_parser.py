"""Unit tests for pdf_parser.py using small PDFs generated with PyMuPDF."""
import fitz
import pytest

from backend.app.ingestion import pdf_parser
from backend.app.ingestion.pdf_parser import (
    OcrUnavailable, canonical_section, parse_pdf)

BODY = ("This is ordinary body text of the paper that goes on for a while so that it "
        "counts as a real paragraph rather than a heading or a caption. " * 2)


def _make_pdf(path, pages):
    """pages: list of list[(text, fontsize, fontname)]"""
    doc = fitz.open()
    for items in pages:
        page = doc.new_page()
        y = 72
        for text, size, font in items:
            page.insert_textbox(fitz.Rect(54, y, 540, y + 120), text, fontsize=size, fontname=font)
            y += 130 if len(text) > 120 else 40
    doc.save(str(path))
    doc.close()


@pytest.fixture
def sectioned_pdf(tmp_path):
    p = tmp_path / "paper.pdf"
    _make_pdf(p, [
        [("1 Introduction", 16, "hebo"), (BODY, 10, "helv")],
        [("2 Method", 16, "hebo"), (BODY, 10, "helv")],
    ])
    return str(p)


def test_parse_text_contains_body(sectioned_pdf):
    assert "ordinary body text" in parse_pdf(sectioned_pdf).text


def test_segments_carry_page_and_section(sectioned_pdf):
    parsed = parse_pdf(sectioned_pdf)
    assert parsed.n_pages == 2 and not parsed.used_ocr
    by_section = {s.section: s.page for s in parsed.segments}
    assert by_section.get("Introduction") == 1
    assert by_section.get("Method") == 2


def test_heading_text_is_not_content(sectioned_pdf):
    assert "1 Introduction" not in parse_pdf(sectioned_pdf).text


def test_segment_offsets_index_full_text(sectioned_pdf):
    parsed = parse_pdf(sectioned_pdf)
    for s in parsed.segments:
        assert parsed.text[s.start:s.end] == s.text


def test_arxiv_watermark_is_ignored(tmp_path):
    p = tmp_path / "w.pdf"
    _make_pdf(p, [[("arXiv:1905.13613v2 [cs.LG] 18 Jun 2020", 20, "hebo"), (BODY, 10, "helv")]])
    parsed = parse_pdf(str(p))
    assert "arXiv:1905" not in parsed.text
    assert all("arxiv" not in s.section.lower() for s in parsed.segments)


def test_canonical_section_names():
    assert canonical_section("3.1 Related Work") == "Related Work"
    assert canonical_section("IV. EXPERIMENTS") == "Experiments"
    assert canonical_section("References") == "References"
    assert canonical_section("Subspace Orthogonalization") == "Subspace Orthogonalization"


def test_ocr_fallback_triggered_for_empty_text_layer(tmp_path, monkeypatch):
    p = tmp_path / "scan.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(p))
    doc.close()

    monkeypatch.setattr(pdf_parser, "_ocr_segments",
                        lambda d: [(1, "Recovered by OCR. " * 10), (2, "Second page text. " * 10)])
    parsed = parse_pdf(str(p))
    assert parsed.used_ocr and "Recovered by OCR" in parsed.text
    assert {s.page for s in parsed.segments} == {1, 2}


def test_ocr_unavailable_raises_clear_error(tmp_path, monkeypatch):
    p = tmp_path / "scan.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(p))
    doc.close()

    def boom(d):
        raise OcrUnavailable("no tesseract")
    monkeypatch.setattr(pdf_parser, "_ocr_segments", boom)
    with pytest.raises(OcrUnavailable):
        parse_pdf(str(p))


def test_normal_pdf_does_not_use_ocr(sectioned_pdf, monkeypatch):
    monkeypatch.setattr(pdf_parser, "_ocr_segments",
                        lambda d: (_ for _ in ()).throw(AssertionError("OCR should not run")))
    assert not parse_pdf(sectioned_pdf).used_ocr


# ── Conclusion must not swallow the appendix ─────────────────────────────────

LONG = ("Additional derivations and supplementary experimental details follow here in full "
        "so that the section runs on well past one page of prose. ") * 3


def _pages(*sections):
    """sections: (heading_or_None, body). One page each."""
    pages = []
    for head, body in sections:
        items = []
        if head:
            items.append((head, 16, "hebo"))
        items.append((body, 10, "helv"))
        pages.append(items)
    return pages


def _sections(tmp_path, *sections):
    p = tmp_path / "x.pdf"
    _make_pdf(p, _pages(*sections))
    return [(s.section, s.text[:20]) for s in parse_pdf(str(p)).segments]


def test_short_conclusion_stays_conclusion(tmp_path):
    secs = _sections(tmp_path, ("1 Introduction", BODY), ("2 Conclusion", BODY))
    assert [s for s, _ in secs] == ["Introduction", "Conclusion"]


def test_unrecognised_heading_after_conclusion_becomes_appendix(tmp_path):
    secs = _sections(tmp_path, ("1 Introduction", BODY), ("2 Conclusion", BODY),
                     ("A Proofs of Theorem One", BODY))
    assert [s for s, _ in secs] == ["Introduction", "Conclusion", "Appendix"]


def test_standard_backmatter_names_survive_after_conclusion(tmp_path):
    secs = _sections(tmp_path, ("1 Introduction", BODY), ("2 Conclusion", BODY),
                     ("Acknowledgments", BODY))
    assert [s for s, _ in secs][-1] == "Acknowledgments"


def test_unheaded_overflow_after_conclusion_becomes_appendix(tmp_path):
    """Appendix text with no recognisable heading must not keep the Conclusion label."""
    p = tmp_path / "x.pdf"
    doc = fitz.open()
    for pg in range(3):                       # ~7k chars of prose after the heading
        page = doc.new_page()
        y = 40
        if pg == 0:
            page.insert_textbox(fitz.Rect(54, 40, 540, 70), "2 Conclusion", fontsize=16, fontname="hebo")
            y = 80
        for _ in range(8):
            page.insert_textbox(fitz.Rect(54, y, 540, y + 90), BODY, fontsize=9, fontname="helv")
            y += 95
    doc.save(str(p)); doc.close()
    segs = parse_pdf(str(p)).segments
    labels = [s.section for s in segs]
    assert labels[0] == "Conclusion" and labels[-1] == "Appendix"
    assert sum(len(s.text) for s in segs if s.section == "Conclusion") <= pdf_parser.CONCLUSION_MAX_CHARS + 1000


def test_heading_before_conclusion_keeps_free_form_name(tmp_path):
    secs = _sections(tmp_path, ("1 Introduction", BODY), ("2 Subspace Orthogonalization", BODY))
    assert secs[1][0] == "Subspace Orthogonalization"
