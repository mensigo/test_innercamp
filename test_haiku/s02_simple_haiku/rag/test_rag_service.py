"""Smoke tests for rag_service endpoints."""

import importlib
from types import ModuleType

import numpy as np
import pytest

from src.s02_simple_haiku.rag import logger as rag_logger

pytestmark = [pytest.mark.unit, pytest.mark.rag]


@pytest.fixture(autouse=True)
def remove_rag_file_sink():
    try:
        rag_logger.logger.remove(rag_logger.FILE_SINK_ID)
    except ValueError:
        pass


class FakeIndex:
    """Minimal FAISS-like index for search tests."""

    def __init__(self, indices: list[int]):
        self.indices = indices

    def search(self, _vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        idx = self.indices[:top_k]
        distances = np.zeros((1, len(idx)), dtype='float32')
        return distances, np.array([idx], dtype='int64')


def load_rag_service(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """
    Load rag_service with patched build_index functions.
    """
    import src.s02_simple_haiku.rag.build_index as bi

    monkeypatch.setattr(bi, 'build_rag_chunks', lambda: [])
    monkeypatch.setattr(bi, 'init_faiss_index', lambda _texts: None)

    module = importlib.import_module('src.s02_simple_haiku.rag.rag_service')
    module = importlib.reload(module)
    monkeypatch.setattr(module, 'init_rag_index', lambda: None)
    module.app.config['RAG_READY'] = True
    return module


def test_health_not_ready(monkeypatch: pytest.MonkeyPatch):
    """Health should be not ready without index."""
    rag_service = load_rag_service(monkeypatch)
    client = rag_service.app.test_client()

    response = client.get('/health')
    data = response.get_json()

    assert response.status_code == 503
    assert data['status'] == 'not_ready'
    assert data['chunks'] == 0


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
    assert data['status'] == 'ok'
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

    # return float64 embeddings to match index dtype
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
    data = response.get_json()

    assert response.status_code == 400
    assert data['error'] == 'Missing question'


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
    data = response.get_json()

    assert response.status_code == 500
    assert data['error'] == 'Search failed'


def test_search_internal_server_error(monkeypatch: pytest.MonkeyPatch):
    """Search should return internal error on generic exception."""
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
        raise Exception('unexpected')

    monkeypatch.setattr(rag_service, 'search_chunks', boom)

    client = rag_service.app.test_client()
    response = client.post('/search', json={'question': 'Q'})
    data = response.get_json()

    assert response.status_code == 500
    assert data['error'] == 'Internal server error'


def test_init_failure_returns_500(monkeypatch: pytest.MonkeyPatch):
    """Initialization failure should return 500."""
    rag_service = load_rag_service(monkeypatch)
    rag_service.app.config['RAG_READY'] = False

    monkeypatch.setattr(
        rag_service,
        'init_rag_index',
        lambda: (_ for _ in ()).throw(RuntimeError('init failed')),
    )

    client = rag_service.app.test_client()
    response = client.post('/search', json={'question': 'Q'})
    data = response.get_json()

    assert response.status_code == 500
    assert data['error'] == 'Initialization failed'
