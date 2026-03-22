# Document Intelligence System

An end-to-end pipeline for structured information extraction from messy, unformatted PDFs using **LayoutLMv3** and **OCR**, with semantic search powered by **FAISS**.

## Architecture

```
Raw PDF
   ↓
[Stage 1/2] PDF Type Detection + OCR
   → Digital PDF  → pdfplumber (text + bbox, no OCR)
   → Scanned PDF  → easyocr   (OCR on rendered images)
   ↓
[Stage 3] LayoutLMv3 Token Classification
   → Labels tokens as HEADER / QUESTION / ANSWER
   ↓
[Stage 4] Post-Processing Layer
   → OCR noise cleaning, date/currency normalisation, confidence filtering
   ↓
[Stage 5/6] Embedding + FAISS Indexing
   → sentence-transformers → cosine similarity search
   ↓
[Stage 7] FastAPI
   → POST /extract   (PDF → structured JSON)
   → POST /search    (query → top-k relevant chunks)
```

## Setup

```bash
pip install -r requirements.txt
```

> Requires Python 3.10+. For GPU inference set `gpu=True` in easyocr and install `faiss-gpu` instead of `faiss-cpu`.

## Usage

### Run the API

```bash
uvicorn src.api.app:app --reload
```

Then visit `http://localhost:8000/docs` for the interactive Swagger UI.

### Process a single PDF (Python)

```python
from src.pipeline import DocumentIntelligencePipeline

pipeline = DocumentIntelligencePipeline.from_config("configs/config.yaml")
doc = pipeline.run("data/raw/invoice.pdf")
print(doc.to_dict())
```

### Semantic search

```python
hits = pipeline.search("invoice total amount due", top_k=5)
for h in hits:
    print(h.score, h.chunk.pdf_name, h.chunk.text)
```

## Dataset

Download and prepare FUNSD:

```bash
python scripts/prepare_funsd.py
```

## Evaluation

Run the baseline vs LayoutLMv3 comparison:

```bash
python scripts/evaluate.py
```

Results are logged to MLflow:

```bash
mlflow ui --backend-store-uri models/mlflow
```

## Project Structure

```
document_intelligence/
├── configs/
│   └── config.yaml             # all tuneable parameters
├── data/
│   ├── raw/                    # input PDFs
│   ├── processed/              # intermediate outputs
│   └── annotations/funsd/      # FUNSD train/test JSON
├── models/
│   ├── checkpoints/            # fine-tuned LayoutLMv3 weights
│   ├── faiss_index.bin         # FAISS index (generated at runtime)
│   ├── faiss_metadata.json     # chunk metadata (generated at runtime)
│   └── mlflow/                 # MLflow tracking
├── notebooks/                  # exploration and fine-tuning notebooks
├── scripts/
│   ├── prepare_funsd.py        # dataset download
│   └── evaluate.py             # baseline vs model F1 evaluation
├── src/
│   ├── ocr/
│   │   └── pdf_loader.py       # Stage 1/2: PDF type detection + token extraction
│   ├── extraction/
│   │   └── layoutlm.py         # Stage 3: LayoutLMv3 token classification
│   ├── postprocessing/
│   │   └── cleaner.py          # Stage 4: OCR cleaning + normalisation
│   ├── embeddings/
│   │   └── indexer.py          # Stage 5/6: FAISS indexing + semantic search
│   ├── api/
│   │   └── app.py              # Stage 7: FastAPI endpoints
│   └── pipeline.py             # End-to-end orchestrator + MLflow logging
├── tests/
│   └── test_postprocessing.py  # Unit tests (no model loading required)
└── requirements.txt
```

## Tech Stack

| Layer | Library |
|---|---|
| Digital PDF parsing | pdfplumber, pymupdf |
| OCR | easyocr, pytesseract |
| Layout model | LayoutLMv3 (HuggingFace transformers) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS (faiss-cpu) |
| Experiment tracking | MLflow |
| API | FastAPI + uvicorn |
| Testing | pytest |
