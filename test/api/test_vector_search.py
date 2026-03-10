from __future__ import annotations

import faiss
import pytest

from src.prepare_data import build_faiss_index
from src.api.vector_search import FAISS_INDEX_PATH, vector_search

pytestmark = [pytest.mark.api]


def test_vector_search_returns_empty_chunks_for_empty_query():
    assert vector_search('', k=2) == {'chunks': []}


def test_vector_search_returns_empty_chunks_for_invalid_k():
    assert vector_search('lecturer', k=0) == {'chunks': []}


def test_vector_search_returns_chunks_shape_for_regular_input():
    build_faiss_index(force=False)
    result = vector_search('машинное обучение', k=2)

    assert isinstance(result, dict)
    assert 'chunks' in result
    assert isinstance(result['chunks'], list)
    assert len(result['chunks']) == 2
    assert all(isinstance(chunk, str) and chunk for chunk in result['chunks'])


def test_vector_search_returns_at_most_available_chunks():
    build_faiss_index(force=False)
    result = vector_search('оптимизация', k=10_000)
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    assert isinstance(result, dict)
    assert 'chunks' in result
    assert isinstance(result['chunks'], list)
    assert len(result['chunks']) == index.ntotal
