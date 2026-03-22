"""
tests/test_postprocessing.py

Unit tests for the OCR cleaning and normalisation logic.
No model loading required — pure function tests.

Run:
    pytest tests/test_postprocessing.py -v
"""

import pytest
from src.postprocessing.cleaner import (
    clean_ocr_text,
    is_noise,
    normalize_date,
    normalize_currency,
    PostProcessor,
    ProcessedDocument,
)
from src.extraction.layoutlm import ExtractionResult, LabeledSpan
from src.ocr.pdf_loader import Token


# ---------------------------------------------------------------------------
# clean_ocr_text
# ---------------------------------------------------------------------------

class TestCleanOCRText:
    def test_collapses_whitespace(self):
        assert clean_ocr_text("hello   world") == "hello world"

    def test_strips_leading_trailing(self):
        assert clean_ocr_text("  foo  ") == "foo"

    def test_replaces_smart_quotes(self):
        result = clean_ocr_text("\u201chello\u201d")
        assert '"' in result

    def test_replaces_em_dash(self):
        assert clean_ocr_text("2020\u20142021") == "2020-2021"

    def test_empty_string(self):
        assert clean_ocr_text("") == ""


# ---------------------------------------------------------------------------
# is_noise
# ---------------------------------------------------------------------------

class TestIsNoise:
    def test_empty_is_noise(self):
        assert is_noise("") is True

    def test_whitespace_is_noise(self):
        assert is_noise("   ") is True

    def test_normal_word_not_noise(self):
        assert is_noise("Invoice") is False

    def test_number_not_noise(self):
        assert is_noise("12345") is False


# ---------------------------------------------------------------------------
# normalize_date
# ---------------------------------------------------------------------------

class TestNormalizeDate:
    @pytest.mark.parametrize("raw, expected", [
        ("01/12/2023", "2023-12-01"),
        ("2023-12-01", "2023-12-01"),
        ("December 01, 2023", "2023-12-01"),
        ("01 December 2023", "2023-12-01"),
    ])
    def test_known_formats(self, raw, expected):
        assert normalize_date(raw) == expected

    def test_non_date_returns_none(self):
        assert normalize_date("not a date") is None

    def test_empty_returns_none(self):
        assert normalize_date("") is None


# ---------------------------------------------------------------------------
# normalize_currency
# ---------------------------------------------------------------------------

class TestNormalizeCurrency:
    def test_usd_symbol(self):
        result = normalize_currency("$1,234.56")
        assert result == {"amount": 1234.56, "currency": "USD"}

    def test_inr_symbol(self):
        result = normalize_currency("₹5000")
        assert result == {"amount": 5000.0, "currency": "INR"}

    def test_word_code(self):
        result = normalize_currency("100 USD")
        assert result["currency"] == "USD"

    def test_no_currency_returns_none(self):
        assert normalize_currency("hello world") is None


# ---------------------------------------------------------------------------
# PostProcessor integration
# ---------------------------------------------------------------------------

def _make_extraction_result(spans: list[LabeledSpan]) -> ExtractionResult:
    return ExtractionResult(pdf_name="test.pdf", pages=1, spans=spans)


def _make_span(text: str, label: str, confidence: float = 0.9) -> LabeledSpan:
    tok = Token(text=text, bbox=(0, 0, 0.1, 0.1), confidence=confidence)
    return LabeledSpan(
        label=label, text=text, tokens=[tok], page=0, avg_confidence=confidence
    )


class TestPostProcessor:
    def test_clean_span_appears_in_fields(self):
        pp = PostProcessor(min_confidence=0.5)
        result = _make_extraction_result([_make_span("Invoice Date", "QUESTION")])
        doc = pp.process(result)
        assert "QUESTION" in doc.fields
        assert doc.fields["QUESTION"][0]["text"] == "Invoice Date"

    def test_low_confidence_span_is_flagged(self):
        pp = PostProcessor(min_confidence=0.5)
        result = _make_extraction_result([_make_span("Blurry text", "ANSWER", confidence=0.3)])
        doc = pp.process(result)
        assert len(doc.flagged) == 1
        assert "ANSWER" not in doc.fields

    def test_noise_span_is_dropped(self):
        pp = PostProcessor(min_confidence=0.5)
        result = _make_extraction_result([_make_span("", "ANSWER")])
        doc = pp.process(result)
        assert len(doc.fields) == 0
        assert len(doc.flagged) == 0

    def test_to_dict_structure(self):
        pp = PostProcessor(min_confidence=0.5)
        result = _make_extraction_result([_make_span("Total: $500", "ANSWER")])
        doc = pp.process(result)
        d = doc.to_dict()
        assert "pdf_name" in d
        assert "fields" in d
        assert "flagged_for_review" in d
