"""Smoke tests for rag_utils helpers."""

import pytest

from src.s02_simple_haiku.rag import rag_utils as ru

pytestmark = pytest.mark.unit


class FakeResponse:
    """Minimal response stub for requests."""

    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self) -> dict:
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError('bad status')


def test_check_health_true(monkeypatch: pytest.MonkeyPatch):
    """Health returns True when index_ready."""
    monkeypatch.setattr(
        ru.requests, 'get', lambda _url, timeout: FakeResponse({'index_ready': True})
    )
    assert ru.check_health() is True


def test_fetch_answer_returns_empty_when_unhealthy(monkeypatch: pytest.MonkeyPatch):
    """Fetch answer returns empty dict when health check fails."""
    monkeypatch.setattr(ru, 'check_health', lambda: False)
    result = ru.fetch_answer('Q')
    assert result == {}


def test_fetch_answer_success(monkeypatch: pytest.MonkeyPatch):
    """Fetch answer returns json on success."""
    monkeypatch.setattr(ru, 'check_health', lambda: True)
    monkeypatch.setattr(
        ru.requests,
        'post',
        lambda _url, json, timeout: FakeResponse({'answer': 'ok'}),
    )
    result = ru.fetch_answer('Q')
    assert result == {'answer': 'ok'}


def test_answer_question_fallback_on_empty(monkeypatch: pytest.MonkeyPatch):
    """Answer question returns fallback on empty response."""
    monkeypatch.setattr(ru, 'fetch_answer', lambda _question: {})
    result = ru.answer_question('Q')
    assert result['answer']
    assert result['chunk_title_list'] == []
    assert result['chunk_texts'] == []


def test_answer_question_error_branch(monkeypatch: pytest.MonkeyPatch):
    """Answer question returns error branch."""
    monkeypatch.setattr(ru, 'fetch_answer', lambda _question: {'error': 'fail'})
    result = ru.answer_question('Q')
    assert result['answer'] == 'fail'
    assert result['chunk_title_list'] == []
    assert result['chunk_texts'] == []


def test_answer_question_success_branch(monkeypatch: pytest.MonkeyPatch):
    """Answer question returns success branch."""
    payload = {
        'answer': 'ok',
        'chunk_title_list': ['T1'],
        'chunk_texts': ['C1'],
    }
    monkeypatch.setattr(ru, 'fetch_answer', lambda _question: payload)
    result = ru.answer_question('Q')
    assert result == payload
