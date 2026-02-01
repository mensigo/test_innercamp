"""Smoke tests for rag_service endpoints."""

import importlib
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class FakeIndex:
    """Minimal FAISS-like index for search tests."""

    def __init__(self, indices: list[int]):
        self.indices = indices

    def search(self, _vectors: np.ndarray, top_k: int):
        idx = self.indices[:top_k]
        return None, np.array([idx], dtype='int64')


def load_rag_service(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """
    Load rag_service with patched build_index functions.
    """
    import src.s02_simple_haiku.rag.build_index as bi

    monkeypatch.setattr(bi, 'build_rag_chunks', lambda: [])
    monkeypatch.setattr(bi, 'init_faiss_index', lambda _texts: (None, 0))

    module = importlib.import_module('src.s02_simple_haiku.rag.rag_service')
    return importlib.reload(module)


def test_health_not_ready(monkeypatch: pytest.MonkeyPatch):
    """Health should be not ready without index."""
    rag_service = load_rag_service(monkeypatch)
    client = rag_service.app.test_client()

    response = client.get('/health')
    data = response.get_json()

    assert response.status_code == 503
    assert data['index_ready'] is False


def test_health_ok(monkeypatch: pytest.MonkeyPatch):
    """Health should be ok when index and chunks exist."""
    rag_service = load_rag_service(monkeypatch)
    rag_service.RAG_CHUNKS = [
        rag_service.RagChunk(
            text='chunk',
            source='doc.md',
            title='Title',
            ordinal=1,
        )
    ]
    rag_service.RAG_INDEX = object()

    client = rag_service.app.test_client()
    response = client.get('/health')
    data = response.get_json()

    assert response.status_code == 200
    assert data['index_ready'] is True
    assert data['chunks'] == 1


def test_search_smoke(monkeypatch: pytest.MonkeyPatch):
    """Search should return answer and chunk metadata."""
    rag_service = load_rag_service(monkeypatch)
    rag_service.RAG_CHUNKS = [
        rag_service.RagChunk(
            text='C1',
            source='a.md',
            title='T1',
            ordinal=1,
        ),
        rag_service.RagChunk(
            text='C2',
            source='b.md',
            title='T2',
            ordinal=1,
        ),
    ]
    rag_service.RAG_INDEX = FakeIndex([1, 0])

    monkeypatch.setattr(rag_service, 'embed_texts', lambda _texts: [[0.1, 0.2, 0.3]])
    monkeypatch.setattr(
        rag_service,
        'post_chat_completions',
        lambda _payload: {'choices': [{'message': {'content': 'ok'}}]},
    )

    client = rag_service.app.test_client()
    response = client.post('/search', json={'question': 'Q'})
    data = response.get_json()

    assert response.status_code == 200
    assert data['answer'] == 'ok'
    assert data['chunk_title_list'] == ['T2', 'T1']
    assert data['chunk_texts'] == ['C2', 'C1']
    assert data['top_k'] == 2


def test_search_missing_question(monkeypatch: pytest.MonkeyPatch):
    """Search should return 400 without question."""
    rag_service = load_rag_service(monkeypatch)
    client = rag_service.app.test_client()

    response = client.post('/search', json={})
    assert response.status_code == 400


def test_search_error_returns_500(monkeypatch: pytest.MonkeyPatch):
    """Search should return 500 on unexpected error."""
    rag_service = load_rag_service(monkeypatch)
    rag_service.RAG_CHUNKS = [
        rag_service.RagChunk(
            text='C1',
            source='a.md',
            title='T1',
            ordinal=1,
        )
    ]
    rag_service.RAG_INDEX = FakeIndex([0])

    def boom(_question: str, _top_k: int):
        raise RuntimeError('boom')

    monkeypatch.setattr(rag_service, 'search_chunks', boom)

    client = rag_service.app.test_client()
    response = client.post('/search', json={'question': 'Q'})
    assert response.status_code == 500
