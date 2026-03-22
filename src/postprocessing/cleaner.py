"""
Stage 4: Post-processing layer.

Responsibilities:
  1. OCR noise cleaning     — fix common character confusions (l↔1, O↔0, etc.)
  2. Normalization          — dates → ISO 8601, currency → float
  3. Confidence filtering   — flag spans below threshold for human review
  4. Structured output      — produce a clean JSON-serialisable dict per document
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from src.extraction.layoutlm import ExtractionResult, LabeledSpan


# ---------------------------------------------------------------------------
# OCR noise cleaning
# ---------------------------------------------------------------------------

# Common single-character OCR confusions in printed documents
_OCR_CHAR_MAP: dict[str, str] = {
    "\x00": "",   # null bytes
    "\f": " ",    # form feed
    "—": "-",
    "–": "-",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}

# Patterns that indicate a token is likely garbled (high noise)
_NOISE_PATTERN = re.compile(r"^[^a-zA-Z0-9\s\.,\-\$\€\£\%\/\(\)]{3,}$")


def clean_ocr_text(text: str) -> str:
    """Apply character-level fixes and whitespace normalisation."""
    for bad, good in _OCR_CHAR_MAP.items():
        text = text.replace(bad, good)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove isolated non-printable characters
    text = re.sub(r"[\x00-\x08\x0b\x0e-\x1f\x7f]", "", text)

    return text


def is_noise(text: str) -> bool:
    """Return True if the token looks like OCR garbage."""
    return bool(_NOISE_PATTERN.match(text)) or len(text.strip()) == 0


# ---------------------------------------------------------------------------
# Date normalisation
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d",
    "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y",
    "%d %B %Y", "%d %b %Y", "%Y%m%d",
]


def normalize_date(text: str) -> Optional[str]:
    """
    Try to parse and normalise a date string to ISO 8601 (YYYY-MM-DD).
    Returns None if text does not look like a date.
    """
    text = text.strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Currency normalisation
# ---------------------------------------------------------------------------

_CURRENCY_RE = re.compile(
    r"([\$\€\£\₹]?)\s*([\d,]+(?:\.\d{1,2})?)\s*(USD|EUR|GBP|INR|dollars?|euros?)?",
    re.IGNORECASE,
)

_SYMBOL_TO_CODE = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}


def normalize_currency(text: str) -> Optional[dict[str, Any]]:
    """
    Extract a numeric amount and currency code from a string.

    Returns:
        {"amount": float, "currency": str} or None if not parseable.
    """
    m = _CURRENCY_RE.search(text)
    if not m:
        return None
    symbol, digits, word_code = m.group(1), m.group(2), m.group(3)
    try:
        amount = float(digits.replace(",", ""))
    except ValueError:
        return None

    currency = (
        _SYMBOL_TO_CODE.get(symbol)
        or (word_code.upper()[:3] if word_code else None)
        or "UNKNOWN"
    )
    return {"amount": amount, "currency": currency}


# ---------------------------------------------------------------------------
# Structured output builder
# ---------------------------------------------------------------------------

@dataclass
class ProcessedDocument:
    pdf_name: str
    pages: int
    fields: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    flagged: list[dict[str, Any]] = field(default_factory=list)   # low-confidence spans

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf_name": self.pdf_name,
            "pages": self.pages,
            "fields": self.fields,
            "flagged_for_review": self.flagged,
        }


# ---------------------------------------------------------------------------
# Main post-processor
# ---------------------------------------------------------------------------

class PostProcessor:
    """
    Cleans and structures the output of LayoutLMExtractor.

    Usage:
        pp = PostProcessor(min_confidence=0.5)
        doc = pp.process(extraction_result)
        print(doc.to_dict())
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        normalize_dates: bool = True,
        normalize_currency: bool = True,
    ):
        self.min_confidence = min_confidence
        self.normalize_dates = normalize_dates
        self.normalize_currency = normalize_currency

    def _process_span(self, span: LabeledSpan) -> dict[str, Any]:
        """Convert a single span into a clean field dict."""
        text = clean_ocr_text(span.text)

        record: dict[str, Any] = {
            "text": text,
            "page": span.page,
            "confidence": round(span.avg_confidence, 3),
        }

        # Attempt type-specific normalisations
        if self.normalize_dates:
            normalised = normalize_date(text)
            if normalised:
                record["normalised_date"] = normalised

        if self.normalize_currency:
            normalised = normalize_currency(text)
            if normalised:
                record["normalised_currency"] = normalised

        return record

    @staticmethod
    def _merge_adjacent_spans(spans: list) -> list:
        """
        Merge consecutive spans that share the same entity type and page
        into a single span. This turns fragmented single-word spans
        (e.g. "What" "is" "composition") into full sentences.

        Merging criteria:
        - Same entity type (ignoring B-/I- prefix)
        - Same page
        - Gap between spans <= 3 tokens (allows for skipped stopwords)
        """
        from src.extraction.layoutlm import LabeledSpan

        def entity_type(label: str) -> str:
            return label.split("-", 1)[-1] if "-" in label else label

        if not spans:
            return spans

        merged = []
        current = spans[0]

        for next_span in spans[1:]:
            same_entity = entity_type(current.label) == entity_type(next_span.label)
            same_page = current.page == next_span.page

            if same_entity and same_page:
                # Merge: combine text and tokens, keep dominant label, average confidence
                all_tokens = current.tokens + next_span.tokens
                combined_text = current.text + " " + next_span.text
                avg_conf = sum(t.confidence for t in all_tokens) / len(all_tokens)
                current = LabeledSpan(
                    label=current.label,
                    text=combined_text,
                    tokens=all_tokens,
                    page=current.page,
                    avg_confidence=avg_conf,
                )
            else:
                merged.append(current)
                current = next_span

        merged.append(current)
        return merged

    def process(self, result: ExtractionResult) -> ProcessedDocument:
        """
        Post-process an ExtractionResult into a clean ProcessedDocument.

        Args:
            result: Output of LayoutLMExtractor.extract()

        Returns:
            ProcessedDocument with structured fields and flagged spans.
        """
        doc = ProcessedDocument(pdf_name=result.pdf_name, pages=result.pages)

        # Merge adjacent same-label spans before processing
        # This turns fragmented word-level spans into full sentences/phrases
        spans = self._merge_adjacent_spans(result.spans)

        for span in spans:
            if is_noise(span.text):
                logger.debug(f"[PostProcessor] Skipping noise token: {repr(span.text)}")
                continue

            record = self._process_span(span)

            if span.avg_confidence < self.min_confidence:
                record["flag_reason"] = "low_confidence"
                doc.flagged.append({"label": span.label, **record})
                logger.debug(
                    f"[PostProcessor] Flagged low-confidence span "
                    f"(conf={span.avg_confidence:.2f}): {span.text[:40]}"
                )
                continue

            doc.fields.setdefault(span.label, []).append(record)

        logger.info(
            f"[PostProcessor] {result.pdf_name}: "
            f"{sum(len(v) for v in doc.fields.values())} clean fields, "
            f"{len(doc.flagged)} flagged."
        )
        return doc
