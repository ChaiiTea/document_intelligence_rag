from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

from src.postprocessing.cleaner import ProcessedDocument


@dataclass
class Chunk:
    text: str
    pdf_name: str
    page: int
    chunk_index: int
    label: Optional[str] = None  


@dataclass
class SearchResult:
    chunk: Chunk
    score: float      

def _chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap: int = 32,
) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start += chunk_size - overlap
    return [c for c in chunks if c]


def document_to_chunks(doc: ProcessedDocument, chunk_size: int = 256, overlap: int = 32) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_idx = 0

    for label, records in doc.fields.items():
        for record in records:
            text = record.get("text", "")
            if not text:
                continue
            for piece in _chunk_text(text, chunk_size, overlap):
                chunks.append(
                    Chunk(
                        text=piece,
                        pdf_name=doc.pdf_name,
                        page=record.get("page", 0),
                        chunk_index=chunk_idx,
                        label=label,
                    )
                )
                chunk_idx += 1

    return chunks


class EmbeddingModel:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"[EmbeddingModel] Loading {self.model_name}...")
        self._model = SentenceTransformer(self.model_name)
        logger.info("[EmbeddingModel] Ready.")

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,   
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")


class FAISSIndex:

    def __init__(self, dim: Optional[int] = None):
        self._index = None
        self._metadata: list[dict[str, Any]] = []
        self._dim = dim

    def _init_index(self, dim: int):
        import faiss
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)   
        logger.info(f"[FAISSIndex] Initialised Flat IP index (dim={dim})")

    def add(self, chunks: list[Chunk], embeddings: np.ndarray):
        if self._index is None:
            self._init_index(embeddings.shape[1])

        self._index.add(embeddings)
        self._metadata.extend([asdict(c) for c in chunks])
        logger.info(f"[FAISSIndex] Added {len(chunks)} vectors. Total: {self._index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        if self._index is None or self._index.ntotal == 0:
            logger.warning("[FAISSIndex] Index is empty.")
            return []

        scores, indices = self._index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append(
                SearchResult(
                    chunk=Chunk(**meta),
                    score=float(score),
                )
            )
        return results

    def save(self, index_path: str, metadata_path: str):
        import faiss
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)
        logger.info(f"[FAISSIndex] Saved index → {index_path}")

    @classmethod
    def load(cls, index_path: str, metadata_path: str) -> "FAISSIndex":
        import faiss
        obj = cls()
        obj._index = faiss.read_index(str(index_path))
        with open(metadata_path) as f:
            obj._metadata = json.load(f)
        logger.info(
            f"[FAISSIndex] Loaded {obj._index.ntotal} vectors from {index_path}"
        )
        return obj


class DocumentIndexer:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 256,
        chunk_overlap: int = 32,
    ):
        self.embedder = EmbeddingModel(embedding_model)
        self.index = FAISSIndex()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def index_document(self, doc: ProcessedDocument):
        chunks = document_to_chunks(doc, self.chunk_size, self.chunk_overlap)
        if not chunks:
            logger.warning(f"[DocumentIndexer] No chunks for {doc.pdf_name}")
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts)
        self.index.add(chunks, embeddings)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_emb = self.embedder.encode([query])
        return self.index.search(query_emb, top_k=top_k)

    def save(self, index_path: str, metadata_path: str):
        self.index.save(index_path, metadata_path)

    def load(self, index_path: str, metadata_path: str):
        self.index = FAISSIndex.load(index_path, metadata_path)
