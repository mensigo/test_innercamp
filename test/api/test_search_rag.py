from __future__ import annotations

import faiss
import pytest

from src.prepare_data import build_faiss_index
from src.api.search_rag import FAISS_INDEX_PATH, search_rag

pytestmark = [pytest.mark.api, pytest.mark.llm]


def test_search_rag_returns_empty_chunks_for_empty_query():
    assert search_rag('', k=2) == {'chunks': []}


def test_search_rag_returns_empty_chunks_for_invalid_k():
    assert search_rag('lecturer', k=0) == {'chunks': []}


def test_search_rag_returns_chunks_shape_for_regular_input():
    build_faiss_index(force=False)
    result = search_rag('машинное обучение', k=2)

    assert isinstance(result, dict)
    assert 'chunks' in result
    assert isinstance(result['chunks'], list)
    assert len(result['chunks']) == 2
    assert all(isinstance(chunk, str) and chunk for chunk in result['chunks'])


def test_search_rag_returns_at_most_available_chunks():
    build_faiss_index(force=False)
    result = search_rag('оптимизация', k=10_000)
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    assert isinstance(result, dict)
    assert 'chunks' in result
    assert isinstance(result['chunks'], list)
    assert len(result['chunks']) == index.ntotal
