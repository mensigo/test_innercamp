from __future__ import annotations

import pytest

from src.api.search_rag import search_rag

pytestmark = [pytest.mark.api, pytest.mark.unit]


def test_search_rag_returns_empty_chunks_for_empty_query():
    assert search_rag('', k=2) == {'chunks': []}


def test_search_rag_returns_empty_chunks_for_invalid_k():
    assert search_rag('lecturer', k=0) == {'chunks': []}


def test_search_rag_returns_chunks_shape_for_regular_input():
    result = search_rag('lecturer', k=2)

    assert isinstance(result, dict)
    assert 'chunks' in result
    assert isinstance(result['chunks'], list)
