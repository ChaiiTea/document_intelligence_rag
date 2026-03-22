"""
Stage 1 & 2: PDF type detection + text/bounding-box extraction.

Routing logic:
  - Digital PDF  → pdfplumber  (no OCR needed, fast, accurate coordinates)
  - Scanned PDF  → easyocr     (OCR on rendered page images)

Returns a list of PageTokens: one per page, each token has text + bbox.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from loguru import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Token:
    text: str
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) normalised 0-1
    confidence: float = 1.0                  # OCR confidence; 1.0 for digital PDFs
    page: int = 0


@dataclass
class PageTokens:
    page_number: int
    width: float
    height: float
    tokens: list[Token] = field(default_factory=list)
    source: Literal["digital", "ocr"] = "digital"


# ---------------------------------------------------------------------------
# PDF type detection
# ---------------------------------------------------------------------------

def _is_digital(pdf_path: Path, text_threshold: int = 20) -> bool:
    """Return True if the PDF has selectable text (not a scanned image)."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:3]:           # sample first 3 pages
                text = page.extract_text() or ""
                if len(text.strip()) >= text_threshold:
                    return True
        return False
    except Exception as e:
        logger.warning(f"pdfplumber detection failed: {e}. Defaulting to OCR.")
        return False


# ---------------------------------------------------------------------------
# Digital PDF extraction (pdfplumber)
# ---------------------------------------------------------------------------

def _extract_digital(pdf_path: Path) -> list[PageTokens]:
    import pdfplumber

    pages: list[PageTokens] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            W, H = page.width, page.height
            page_tokens = PageTokens(
                page_number=i,
                width=W,
                height=H,
                source="digital",
            )
            words = page.extract_words(
                x_tolerance=3, y_tolerance=3, keep_blank_chars=False
            )
            for w in words:
                page_tokens.tokens.append(
                    Token(
                        text=w["text"],
                        bbox=(
                            w["x0"] / W,
                            w["top"] / H,
                            w["x1"] / W,
                            w["bottom"] / H,
                        ),
                        confidence=1.0,
                        page=i,
                    )
                )
            pages.append(page_tokens)
    return pages


# ---------------------------------------------------------------------------
# Scanned PDF extraction (easyocr via rendered page images)
# ---------------------------------------------------------------------------

def _extract_scanned(pdf_path: Path, dpi: int = 300) -> list[PageTokens]:
    import fitz          # pymupdf
    import easyocr
    import numpy as np

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    pages: list[PageTokens] = []

    doc = fitz.open(str(pdf_path))
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)   # scale to target DPI
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        W, H = pix.width, pix.height
        page_tokens = PageTokens(
            page_number=i, width=W, height=H, source="ocr"
        )

        results = reader.readtext(img)           # [[bbox, text, conf], ...]
        for bbox_pts, text, conf in results:
            # easyocr returns 4 corner points; convert to (x0,y0,x1,y1)
            xs = [p[0] for p in bbox_pts]
            ys = [p[1] for p in bbox_pts]
            page_tokens.tokens.append(
                Token(
                    text=text,
                    bbox=(
                        min(xs) / W,
                        min(ys) / H,
                        max(xs) / W,
                        max(ys) / H,
                    ),
                    confidence=conf,
                    page=i,
                )
            )
        pages.append(page_tokens)

    doc.close()
    return pages


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def load_pdf(pdf_path: str | Path, dpi: int = 300) -> list[PageTokens]:
    """
    Load a PDF and return token-level data with bounding boxes.

    Args:
        pdf_path: Path to the PDF file.
        dpi:      Resolution used when rendering scanned pages.

    Returns:
        List of PageTokens, one per page.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if _is_digital(pdf_path):
        logger.info(f"[load_pdf] Digital PDF detected → pdfplumber: {pdf_path.name}")
        return _extract_digital(pdf_path)
    else:
        logger.info(f"[load_pdf] Scanned PDF detected → easyocr: {pdf_path.name}")
        return _extract_scanned(pdf_path, dpi=dpi)
