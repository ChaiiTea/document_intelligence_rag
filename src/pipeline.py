from __future__ import annotations
import time
from pathlib import Path
from typing import Optional
import mlflow
from loguru import logger
from src.ocr.pdf_loader import load_pdf
from src.extraction.layoutlm import LayoutLMExtractor
from src.postprocessing.cleaner import PostProcessor, ProcessedDocument
from src.embeddings.indexer import DocumentIndexer


class DocumentIntelligencePipeline:
    def __init__(
        self,
        layoutlm_checkpoint: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        min_confidence: float = 0.5,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        faiss_index_path: str = "models/faiss_index.bin",
        faiss_metadata_path: str = "models/faiss_metadata.json",
        mlflow_tracking_uri: str = "models/mlflow",
        mlflow_experiment: str = "document-intelligence",
        ocr_dpi: int = 300,
    ):
        self.faiss_index_path = faiss_index_path
        self.faiss_metadata_path = faiss_metadata_path
        self.ocr_dpi = ocr_dpi

        self.extractor = LayoutLMExtractor(checkpoint=layoutlm_checkpoint)
        self.postprocessor = PostProcessor(
            min_confidence=min_confidence,
            normalize_dates=True,
            normalize_currency=True,
        )
        self.indexer = DocumentIndexer(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment)

        if Path(faiss_index_path).exists() and Path(faiss_metadata_path).exists():
            self.indexer.load(faiss_index_path, faiss_metadata_path)

    @classmethod
    def from_config(cls, config_path: str = "configs/config.yaml") -> "DocumentIntelligencePipeline":
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        return cls(
            layoutlm_checkpoint=cfg["layoutlm"].get("fine_tuned_checkpoint"),
            embedding_model=cfg["embeddings"]["model_name"],
            min_confidence=cfg["postprocessing"]["min_field_confidence"],
            chunk_size=cfg["embeddings"]["chunk_size"],
            chunk_overlap=cfg["embeddings"]["chunk_overlap"],
            faiss_index_path=cfg["paths"]["faiss_index"],
            faiss_metadata_path=cfg["paths"]["faiss_metadata"],
            mlflow_tracking_uri=cfg["mlflow"]["tracking_uri"],
            mlflow_experiment=cfg["mlflow"]["experiment_name"],
            ocr_dpi=cfg["ocr"]["dpi"],
        )

    def run(self, pdf_path: str, log_to_mlflow: bool = True) -> ProcessedDocument:
        pdf_path = Path(pdf_path)
        logger.info(f"[Pipeline] ▶ Starting: {pdf_path.name}")
        t0 = time.perf_counter()

        with mlflow.start_run(run_name=pdf_path.name) as run:
            mlflow.log_param("pdf_name", pdf_path.name)

            pages = load_pdf(pdf_path, dpi=self.ocr_dpi)
            mlflow.log_metric("pages", len(pages))
            total_tokens = sum(len(p.tokens) for p in pages)
            mlflow.log_metric("total_tokens", total_tokens)

            extraction = self.extractor.extract(pages, pdf_name=pdf_path.name)
            mlflow.log_metric("spans_extracted", len(extraction.spans))

            doc = self.postprocessor.process(extraction)
            clean_fields = sum(len(v) for v in doc.fields.values())
            mlflow.log_metric("clean_fields", clean_fields)
            mlflow.log_metric("flagged_spans", len(doc.flagged))

            self.indexer.index_document(doc)
            self.indexer.save(self.faiss_index_path, self.faiss_metadata_path)

            elapsed = time.perf_counter() - t0
            mlflow.log_metric("processing_time_s", round(elapsed, 2))

            logger.info(
                f"[Pipeline] ✓ {pdf_path.name} | "
                f"{len(pages)} pages | {clean_fields} fields | "
                f"{len(doc.flagged)} flagged | {elapsed:.2f}s"
            )

        return doc

    def run_batch(self, pdf_dir: str) -> list[ProcessedDocument]:
        pdfs = list(Path(pdf_dir).glob("*.pdf"))
        logger.info(f"[Pipeline] Batch: {len(pdfs)} PDFs in {pdf_dir}")
        return [self.run(p) for p in pdfs]

    def search(self, query: str, top_k: int = 5):
        return self.indexer.search(query, top_k=top_k)
