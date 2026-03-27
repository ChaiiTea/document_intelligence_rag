from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from loguru import logger


@dataclass
class Token:
    text: str
    bbox: tuple[float, float, float, float]  
    confidence: float = 1.0                  
    page: int = 0


@dataclass
class PageTokens:
    page_number: int
    width: float
    height: float
    tokens: list[Token] = field(default_factory=list)
    source: Literal["digital", "ocr"] = "digital"


def _is_digital(pdf_path: Path, text_threshold: int = 20) -> bool:
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:3]:           
                text = page.extract_text() or ""
                if len(text.strip()) >= text_threshold:
                    return True
        return False
    except Exception as e:
        logger.warning(f"pdfplumber detection failed: {e}. Defaulting to OCR.")
        return False


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


def _extract_scanned(pdf_path: Path, dpi: int = 300) -> list[PageTokens]:
    import fitz          
    import easyocr
    import numpy as np

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    pages: list[PageTokens] = []

    doc = fitz.open(str(pdf_path))
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)   
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        W, H = pix.width, pix.height
        page_tokens = PageTokens(
            page_number=i, width=W, height=H, source="ocr"
        )

        results = reader.readtext(img)           
        for bbox_pts, text, conf in results:
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


def load_pdf(pdf_path: str | Path, dpi: int = 300) -> list[PageTokens]:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if _is_digital(pdf_path):
        logger.info(f"[load_pdf] Digital PDF detected → pdfplumber: {pdf_path.name}")
        return _extract_digital(pdf_path)
    else:
        logger.info(f"[load_pdf] Scanned PDF detected → easyocr: {pdf_path.name}")
        return _extract_scanned(pdf_path, dpi=dpi)
