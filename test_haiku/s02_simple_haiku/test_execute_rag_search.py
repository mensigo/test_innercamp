"""Smoke tests for rag_utils helpers."""

import pytest
import requests

from src.s02_simple_haiku import execute_rag_search as ers

pytestmark = [pytest.mark.unit, pytest.mark.rag]


class FakeResponse:
    """Minimal response stub for requests."""

    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self) -> dict:
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'HTTP Error: {self.status_code}')


def test_check_health_true(monkeypatch: pytest.MonkeyPatch):
    """Health returns True when service reports ok."""
    monkeypatch.setattr(
        ers.requests, 'get', lambda _url, timeout: FakeResponse({'status': 'ok'})
    )
    assert ers.check_health() is True


def test_check_health_false_on_error(monkeypatch: pytest.MonkeyPatch):
    """Health returns False on http error."""
    monkeypatch.setattr(
        ers.requests, 'get', lambda _url, timeout: FakeResponse({}, status_code=500)
    )
    assert ers.check_health() is False


def test_answer_question_returns_error_on_unhealthy(monkeypatch: pytest.MonkeyPatch):
    """answer_question returns error dict when health check fails."""
    monkeypatch.setattr(ers, 'check_health', lambda _timeout=5: False)
    result = ers.answer_question('Q')
    assert result == {'error': 'Health check failed'}


def test_answer_question_handles_search_error(monkeypatch: pytest.MonkeyPatch):
    """answer_question wraps search error message."""
    monkeypatch.setattr(ers, 'check_health', lambda _timeout=5: True)
    monkeypatch.setattr(
        ers.requests,
        'post',
        lambda _url, json, timeout: FakeResponse({'error': 'boom'}),
    )
    result = ers.answer_question('Q', top_k=3)
    assert result == {'error': 'Search error: boom'}


def test_answer_question_success(monkeypatch: pytest.MonkeyPatch):
    """answer_question returns payload on success."""
    payload = {
        'answer': 'ok',
        'chunk_title_list': ['T1'],
        'chunk_texts': ['C1'],
    }
    monkeypatch.setattr(ers, 'check_health', lambda _timeout=5: True)
    monkeypatch.setattr(
        ers.requests, 'post', lambda _url, json, timeout: FakeResponse(payload)
    )
    result = ers.answer_question('Q')
    assert result == payload


def test_answer_question_handles_unexpected_error(monkeypatch: pytest.MonkeyPatch):
    """answer_question returns unexpected error message on exception."""
    monkeypatch.setattr(ers, 'check_health', lambda _timeout=5: True)
    monkeypatch.setattr(
        ers.requests,
        'post',
        lambda _url, json, timeout: (_ for _ in ()).throw(TimeoutError('fail')),
    )
    result = ers.answer_question('Q')
    assert 'Unexpected error' in result['error']
    assert 'fail' in result['error']
