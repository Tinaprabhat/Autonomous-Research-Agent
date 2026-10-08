"""
pdf_parser.py — PDF -> provenance-carrying text segments.

`parse_pdf()` returns a `ParsedPdf` whose `segments` are paragraph-sized blocks,
each tagged with the page it came from, the section heading it falls under and
its character span inside `ParsedPdf.text`. The chunker builds on those tags so
every chunk carries (page, section, char_start, char_end).

Section detection is heuristic (PyMuPDF span data): a short line that is larger
than body text or bold, or that matches a known section name / numbered-heading
pattern, starts a new section. If a PDF has (almost) no text layer, we fall back
to OCR (pytesseract over rasterised pages).
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

import fitz  # PyMuPDF

log = logging.getLogger(__name__)

# A page with fewer extractable characters than this is treated as "no text layer".
OCR_MIN_CHARS_PER_PAGE = 40

_KNOWN_SECTIONS = [
    ("abstract", "Abstract"),
    ("introduction", "Introduction"),
    ("related work", "Related Work"),
    ("background", "Background"),
    ("preliminar", "Background"),
    ("method", "Method"),
    ("approach", "Method"),
    ("model", "Method"),
    ("experiment", "Experiments"),
    ("evaluation", "Experiments"),
    ("result", "Results"),
    ("discussion", "Discussion"),
    ("conclusion", "Conclusion"),
    ("limitation", "Discussion"),
    ("reference", "References"),
    ("bibliography", "References"),
    ("appendix", "Appendix"),
    ("acknowledg", "Acknowledgments"),
]
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,2}\.?|[A-Z]\.|[IVX]{1,5}\.)\s+[A-Z][^.!?]{1,80}$")
# A Conclusion is about a page of prose. Text beyond this with no new detected heading is almost
# always an appendix / supplementary material whose heading style we failed to recognise.
CONCLUSION_MAX_CHARS = 3500
_CANONICAL_NAMES = {name for _, name in _KNOWN_SECTIONS}
_ARXIV_STAMP =re.compile(r"^arxiv:\s*\d", re.I)
_LEADING_NUM =re.compile(r"^(\d{1,2}(\.\d{1,2}){0,2}\.?|[IVX]{1,5}\.)\s+")


class OcrUnavailable(RuntimeError):
    """PDF needs OCR but pytesseract / the tesseract binary is not available."""


@dataclass
class Segment:
    text: str
    page: int | None
    section: str
    start: int = 0
    end: int = 0


@dataclass
class ParsedPdf:
    text: str
    segments: list[Segment]
    n_pages: int
    used_ocr: bool = False
    meta: dict = field(default_factory=dict)


def canonical_section(heading: str) -> str:
    h = _LEADING_NUM.sub("", heading.strip()).strip(" .:").lower()
    for key, name in _KNOWN_SECTIONS:
        if h.startswith(key) or (len(h) < 40 and key in h and key in ("abstract", "reference", "conclusion", "appendix")):
            return name
    cleaned = _LEADING_NUM.sub("", heading.strip()).strip(" .:")
    return cleaned[:60].title() if cleaned else "Unknown"


def _clean(text: str) -> str:
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    text = text.replace("\r", "").replace("\t", " ").replace("\x00", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)        # de-hyphenate line wraps
    text = re.sub(r"[ ]{2,}", " ", text)
    return text


def _is_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
    t = text.strip()
    if not t or len(t) > 90 or len(t.split()) > 12:
        return False
    if t.endswith((".", ",", ";")) and not _NUMBERED_HEADING.match(t):
        return False
    low = _LEADING_NUM.sub("", t).strip(" .:").lower()
    known = any(low == k or low.startswith(k + " ") for k, _ in _KNOWN_SECTIONS)
    if known and (bold or size >= body_size * 1.05 or _NUMBERED_HEADING.match(t) or t.isupper()):
        return True
    if _NUMBERED_HEADING.match(t) and (bold or size >= body_size * 1.05):
        return True
    return size >= body_size * 1.25 and len(t) > 3


def _body_font_size(doc: fitz.Document) -> float:
    counts: Counter = Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    counts[round(s["size"], 1)] += len(s["text"])
    return counts.most_common(1)[0][0] if counts else 10.0


def _block_text_and_style(block: dict) -> tuple[str, float, bool]:
    lines, sizes, bolds, weights = [], [], [], []
    for line in block.get("lines", []):
        txt = "".join(s["text"] for s in line["spans"]).strip()
        if not txt:
            continue
        lines.append(txt)
        for s in line["spans"]:
            n = len(s["text"])
            sizes.append(s["size"] * n)
            weights.append(n)
            bolds.append(bool(s["flags"] & 16) or "bold" in s["font"].lower())
    if not lines:
        return "", 0.0, False
    total = max(sum(weights), 1)
    return "\n".join(lines), sum(sizes) / total, all(bolds)


def _ocr_segments(doc: fitz.Document) -> list[tuple[int, str]]:
    try:
        import pytesseract
        from PIL import Image
        pytesseract.get_tesseract_version()
    except Exception as e:  # missing module or missing binary
        raise OcrUnavailable(
            "PDF has no text layer and OCR is unavailable "
            f"(install Tesseract + pytesseract): {e}") from e
    out = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        out.append((i, pytesseract.image_to_string(img)))
    return out


def parse_pdf(pdf_path: str) -> ParsedPdf:
    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    try:
        body = _body_font_size(doc)
        raw: list[tuple[int, str, float, bool]] = []  # (page, text, size, bold)
        total_chars = 0
        for pno, page in enumerate(doc, start=1):
            for b in page.get_text("dict")["blocks"]:
                if b.get("type") != 0:
                    continue
                text, size, bold = _block_text_and_style(b)
                text = _clean(text).strip()
                if text:
                    raw.append((pno, text, size, bold))
                    total_chars += len(text)

        used_ocr = False
        if n_pages and total_chars / n_pages < OCR_MIN_CHARS_PER_PAGE:
            log.info("%s: near-empty text layer, falling back to OCR", pdf_path)
            raw = []
            for pno, txt in _ocr_segments(doc):
                for para in re.split(r"\n\s*\n", _clean(txt)):
                    para = " ".join(para.split())
                    if para:
                        raw.append((pno, para, body, False))
            used_ocr = True
    finally:
        doc.close()

    segments: list[Segment] = []
    section = "Unknown"
    cursor = 0
    parts: list[str] = []
    after_conclusion = False     # a Conclusion heading has been seen
    concl_chars = 0              # prose accumulated under the current Conclusion
    for pno, text, size, bold in raw:
        flat = " ".join(text.split())
        if _ARXIV_STAMP.match(flat):
            continue  # arXiv sidebar watermark, not content
        if not used_ocr and _is_heading(flat, size, bold, body):
            section = canonical_section(flat)
            # After the conclusion only the standard back-matter names are meaningful;
            # any other heading (proofs, extra experiments, ...) is appendix material.
            if after_conclusion and section not in _CANONICAL_NAMES:
                section = "Appendix"
            if section == "Conclusion":
                after_conclusion, concl_chars = True, 0
            continue  # headings label following text; not content themselves
        if section == "Conclusion":
            concl_chars += len(flat)
            if concl_chars > CONCLUSION_MAX_CHARS:
                section = "Appendix"   # undetected appendix heading
        seg =Segment(text=flat, page=pno, section=section, start=cursor, end=cursor + len(flat))
        segments.append(seg)
        parts.append(flat)
        cursor = seg.end + 2  # "\n\n" separator
    return ParsedPdf(text="\n\n".join(parts), segments=segments,
                     n_pages=n_pages, used_ocr=used_ocr)


# [DEAD CODE] `extract_text_from_pdf` has no callers (checked 2026-10-08); commented out, restore if needed.
# def extract_text_from_pdf(pdf_path: str) -> str:
#     """Plain-text convenience wrapper (kept for callers that only need text)."""
#     return parse_pdf(pdf_path).text
