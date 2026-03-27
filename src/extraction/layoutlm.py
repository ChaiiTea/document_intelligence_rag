from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
 
from loguru import logger
 
from src.ocr.pdf_loader import PageTokens, Token
 
 
@dataclass
class LabeledSpan:
    label: str
    text: str
    tokens: list[Token]
    page: int
    avg_confidence: float
 
 
@dataclass
class ExtractionResult:
    pdf_name: str
    pages: int
    spans: list[LabeledSpan]
 
    def get_spans_by_label(self, label: str) -> list[LabeledSpan]:
        return [s for s in self.spans if s.label == label]
 
 
class LayoutLMExtractor:
    def __init__(self, checkpoint: Optional[str] = None):
        self._model = None
        self._processor = None
        self._checkpoint = checkpoint or "nielsr/layoutlmv3-finetuned-funsd"
        logger.info(f"[LayoutLMExtractor] Will load checkpoint: {self._checkpoint}")
 
    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor, AutoModelForTokenClassification
 
        logger.info("[LayoutLMExtractor] Loading model...")
        self._processor = AutoProcessor.from_pretrained(
            self._checkpoint, apply_ocr=False
        )
        self._model = AutoModelForTokenClassification.from_pretrained(
            self._checkpoint
        )
        self._model.eval()
        logger.info("[LayoutLMExtractor] Model loaded.")
 
    @staticmethod
    def _normalise_bbox(bbox: tuple, scale: int = 1000) -> list[int]:
        x0, y0, x1, y1 = bbox
        return [
            max(0, min(1000, int(x0 * scale))),
            max(0, min(1000, int(y0 * scale))),
            max(0, min(1000, int(x1 * scale))),
            max(0, min(1000, int(y1 * scale))),
        ]
 
    def _run_page(self, page: PageTokens) -> list[tuple[Token, str]]:
        import torch
        from PIL import Image
 
        if not page.tokens:
            return []
 
        dummy_image = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
 
        chunk_size = 480
        results: list[tuple[Token, str]] = []
 
        for start in range(0, len(page.tokens), chunk_size):
            chunk_tokens = page.tokens[start: start + chunk_size]
            words = [t.text for t in chunk_tokens]
            boxes = [self._normalise_bbox(t.bbox) for t in chunk_tokens]
 
            encoding = self._processor(
                images=dummy_image,
                text=words,
                boxes=boxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512,
            )
 
            with torch.no_grad():
                outputs = self._model(**encoding)
 
            predictions = outputs.logits.argmax(-1).squeeze(0).tolist()
 
            word_ids = encoding.word_ids(batch_index=0)
            seen: set[int] = set()
 
            for idx, word_id in enumerate(word_ids):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                if word_id >= len(chunk_tokens):
                    continue
                label_id = predictions[idx]
                label = self._model.config.id2label.get(label_id, "O")
                results.append((chunk_tokens[word_id], label))
 
        return results
 
    @staticmethod
    def _group_into_spans(
        token_labels: list[tuple[Token, str]], page: int
    ) -> list[LabeledSpan]:
        spans: list[LabeledSpan] = []
        current_tokens: list[Token] = []
        current_label: Optional[str] = None
 
        def flush():
            if current_tokens and current_label and current_label != "O":
                spans.append(LabeledSpan(
                    label=current_label,
                    text=" ".join(t.text for t in current_tokens),
                    tokens=list(current_tokens),
                    page=page,
                    avg_confidence=sum(t.confidence for t in current_tokens)
                    / len(current_tokens),
                ))
 
        for token, raw_label in token_labels:
            entity = raw_label.split("-", 1)[-1] if "-" in raw_label else raw_label
            current_entity = current_label.split("-", 1)[-1] if current_label and "-" in current_label else current_label
            if raw_label.startswith("B-") or entity != current_entity:
                flush()
                current_tokens = [token]
                current_label = raw_label   
            else:
                current_label = raw_label
                current_tokens.append(token)
 
        flush()
        return spans
 
    def extract(
        self, pages: list[PageTokens], pdf_name: str = "document.pdf"
    ) -> ExtractionResult:
        self._load()
 
        all_spans: list[LabeledSpan] = []
        for page in pages:
            logger.debug(f"[extract] Page {page.page_number}: {len(page.tokens)} tokens")
            token_labels = self._run_page(page)
            spans = self._group_into_spans(token_labels, page=page.page_number)
            all_spans.extend(spans)
 
        logger.info(f"[extract] {pdf_name}: {len(all_spans)} spans across {len(pages)} pages")
        return ExtractionResult(pdf_name=pdf_name, pages=len(pages), spans=all_spans)