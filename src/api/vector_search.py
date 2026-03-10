"""API function placeholder for vector search."""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.utils import get_embeddings

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
COURSES_DIR = DATA_DIR / 'courses'
FAISS_INDEX_PATH = COURSES_DIR / 'faiss.index'
RAG_CHUNKS_PATH = COURSES_DIR / 'rag_chunks.json'


def _load_chunks() -> list[str]:
    """Load stored RAG chunks."""
    if not RAG_CHUNKS_PATH.exists():
        raise RuntimeError(
            'vector_search // chunks file is missing, run src/prepare_data.py first'
        )
    try:
        payload = json.loads(RAG_CHUNKS_PATH.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError):
        raise RuntimeError('vector_search // failed to read chunks metadata')

    if not isinstance(payload, dict):
        raise RuntimeError('vector_search // chunks metadata must be a JSON object')

    raw_chunks = payload.get('chunks', [])
    if not isinstance(raw_chunks, list):
        raise RuntimeError('vector_search // chunks metadata has invalid "chunks" type')
    if not raw_chunks:
        raise RuntimeError('vector_search // chunks list is empty')

    chunks = [chunk for chunk in raw_chunks if isinstance(chunk, str)]
    if len(chunks) != len(raw_chunks):
        raise RuntimeError(
            'vector_search // chunks metadata contains non-string entries'
        )

    return chunks


def _embed_query(query: str) -> list[float]:
    """Получить эмбеддинг для текста запроса."""
    try:
        return get_embeddings({'input': query})
    except Exception as ex:
        raise RuntimeError(
            'vector_search // cannot extract embedding from response'
        ) from ex


def vector_search(query: str, k: int = 2) -> dict[str, list[str]]:
    """Return top-k chunks for a query from RAG index."""
    if not query:
        return {'chunks': []}
    if k < 1:
        return {'chunks': []}

    if not FAISS_INDEX_PATH.exists():
        raise RuntimeError(
            'vector_search // FAISS index is missing, run src/prepare_data.py first'
        )

    chunks = _load_chunks()

    try:
        index = faiss.read_index(str(FAISS_INDEX_PATH))
    except RuntimeError as exc:
        raise RuntimeError('vector_search // failed to load FAISS index') from exc

    embedding = _embed_query(query)
    if index.d != len(embedding):
        raise RuntimeError(
            'vector_search // embedding dimension mismatch with FAISS index'
        )
    if index.ntotal != len(chunks):
        raise RuntimeError(
            'vector_search // FAISS vectors count does not match chunks count'
        )

    query_vector = np.array([embedding], dtype=np.float32)
    top_k = min(k, len(chunks))
    _, indices = index.search(query_vector, top_k)

    result_chunks = [chunks[int(idx)] for idx in indices[0]]
    return {'chunks': result_chunks}
