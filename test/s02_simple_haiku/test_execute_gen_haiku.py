"""Tests for execute_gen_haiku client."""

import pytest
import requests

from src.s02_simple_haiku import execute_gen_haiku as eg

pytestmark = [pytest.mark.unit, pytest.mark.haiku]


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
    """Health returns True when status is ok."""
    monkeypatch.setattr(
        eg.requests, 'get', lambda _url, timeout: FakeResponse({'status': 'ok'})
    )
    assert eg.check_health() is True


def test_check_health_false_on_bad_status(monkeypatch: pytest.MonkeyPatch):
    """Health returns False when status is not ok."""
    monkeypatch.setattr(
        eg.requests, 'get', lambda _url, timeout: FakeResponse({'status': 'error'})
    )
    assert eg.check_health() is False


def test_check_health_false_on_connection_error(monkeypatch: pytest.MonkeyPatch):
    """Health returns False on connection error."""

    def raise_connection_error(_url, timeout):
        raise eg.requests.exceptions.ConnectionError('Connection refused')

    monkeypatch.setattr(eg.requests, 'get', raise_connection_error)
    assert eg.check_health() is False


def test_check_health_false_on_timeout(monkeypatch: pytest.MonkeyPatch):
    """Health returns False on timeout."""

    def raise_timeout(_url, timeout):
        raise eg.requests.exceptions.Timeout('Timeout')

    monkeypatch.setattr(eg.requests, 'get', raise_timeout)
    assert eg.check_health() is False


def test_generate_haiku_returns_error_when_unhealthy(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku returns error dict when health check fails."""
    monkeypatch.setattr(eg, 'check_health', lambda _timeout=5: False)
    result = eg.generate_haiku('тема')
    assert result == {'error': 'Health check failed'}


def test_generate_haiku_success(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku returns json on success."""
    monkeypatch.setattr(eg, 'check_health', lambda _timeout=5: True)
    monkeypatch.setattr(
        eg.requests,
        'post',
        lambda _url, json, timeout: FakeResponse(
            {
                'haiku_text': 'Ветер шепчет мне\nО тайнах древних времен\nЛистья кружатся',
                'syllables_per_line': [5, 7, 5],
                'total_words': 9,
                'theme': 'осень',
            }
        ),
    )
    result = eg.generate_haiku('осень')
    assert result['haiku_text']
    assert result['syllables_per_line'] == [5, 7, 5]
    assert result['total_words'] == 9
    assert result['theme'] == 'осень'


def test_generate_haiku_service_error(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku returns error when service responds with error field."""
    monkeypatch.setattr(eg, 'check_health', lambda _timeout=5: True)
    monkeypatch.setattr(
        eg.requests, 'post', lambda _url, json, timeout: FakeResponse({'error': 'fail'})
    )
    result = eg.generate_haiku('тема')
    assert result['error'] == 'Generation error: fail'


def test_generate_haiku_unexpected_exception(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku returns error dict on unexpected exception."""
    monkeypatch.setattr(eg, 'check_health', lambda _timeout=5: True)

    def raise_generic_exception(_url, json, timeout):
        raise ValueError('Unexpected error')

    monkeypatch.setattr(eg.requests, 'post', raise_generic_exception)
    result = eg.generate_haiku('тема')
    assert result['error'] == 'Unexpected error: Unexpected error'


def test_generate_haiku_http_error(monkeypatch: pytest.MonkeyPatch):
    """HTTP error from service bubbles into unexpected error response."""
    monkeypatch.setattr(eg, 'check_health', lambda _timeout=5: True)

    def http_error_response(_url, json, timeout):
        return FakeResponse({'error': 'fail'}, status_code=500)

    monkeypatch.setattr(eg.requests, 'post', http_error_response)
    result = eg.generate_haiku('тема')
    assert 'Unexpected error' in result['error']


def test_generate_haiku_invalid_json(monkeypatch: pytest.MonkeyPatch):
    """Malformed JSON triggers unexpected error path."""
    monkeypatch.setattr(eg, 'check_health', lambda _timeout=5: True)

    class BadJsonResponse(FakeResponse):
        def json(self) -> dict:
            raise ValueError('Invalid JSON')

    monkeypatch.setattr(
        eg.requests, 'post', lambda _url, json, timeout: BadJsonResponse({}, 200)
    )
    result = eg.generate_haiku('тема')
    assert result['error'] == 'Unexpected error: Invalid JSON'
