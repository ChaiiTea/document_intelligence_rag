"""Quick diagnostic to see what model_predict actually returns."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.ocr.pdf_loader import PageTokens, Token
from src.extraction.layoutlm import LayoutLMExtractor

# Load one test example
with open("data/annotations/funsd/test.json") as f:
    examples = json.load(f)

ex = examples[0]
if "words" not in ex or not ex["words"]:
    ex["words"] = ex.get("tokens", [])

print(f"Document: {ex.get('id', '')}")
print(f"Words: {len(ex['words'])}")
print(f"True labels (first 10): {ex['labels'][:10]}")
print(f"First few words: {ex['words'][:5]}")
print(f"First few bboxes: {ex['bboxes'][:5]}")

# Build tokens
tokens = []
for word, bbox in zip(ex["words"], ex["bboxes"]):
    tokens.append(Token(
        text=word,
        bbox=(bbox[0]/1000, bbox[1]/1000, bbox[2]/1000, bbox[3]/1000),
        confidence=1.0,
        page=0,
    ))

page = PageTokens(page_number=0, width=1000, height=1000, tokens=tokens)
extractor = LayoutLMExtractor()
result = extractor.extract([page], pdf_name="debug")

print(f"\nSpans extracted: {len(result.spans)}")
for span in result.spans[:5]:
    print(f"  Label={span.label}, Text='{span.text[:40]}', Tokens={len(span.tokens)}")
    if span.tokens:
        tok = span.tokens[0]
        print(f"    First token id: {id(tok)}, text: {tok.text}")

print(f"\nOriginal token ids (first 5):")
for i, t in enumerate(tokens[:5]):
    print(f"  [{i}] id={id(t)}, text={t.text}")

# Check if span tokens are same objects
token_id_to_idx = {id(t): i for i, t in enumerate(tokens)}
matched = 0
for span in result.spans:
    for tok in span.tokens:
        if id(tok) in token_id_to_idx:
            matched += 1

print(f"\nMatched tokens via id(): {matched} / {sum(len(s.tokens) for s in result.spans)}")
