"""Smoke tests for haiku_utils helpers."""

import pytest

from src.s02_simple_haiku.haiku import haiku_utils as hu

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
    """Health returns True when status is ok."""
    monkeypatch.setattr(
        hu.requests, 'get', lambda _url, timeout: FakeResponse({'status': 'ok'})
    )
    assert hu.check_health() is True


def test_check_health_false_on_bad_status(monkeypatch: pytest.MonkeyPatch):
    """Health returns False when status is not ok."""
    monkeypatch.setattr(
        hu.requests, 'get', lambda _url, timeout: FakeResponse({'status': 'error'})
    )
    assert hu.check_health() is False


def test_check_health_false_on_connection_error(monkeypatch: pytest.MonkeyPatch):
    """Health returns False on connection error."""

    def raise_connection_error(_url, timeout):
        raise hu.requests.exceptions.ConnectionError('Connection refused')

    monkeypatch.setattr(hu.requests, 'get', raise_connection_error)
    assert hu.check_health() is False


def test_check_health_false_on_timeout(monkeypatch: pytest.MonkeyPatch):
    """Health returns False on timeout."""

    def raise_timeout(_url, timeout):
        raise hu.requests.exceptions.Timeout('Timeout')

    monkeypatch.setattr(hu.requests, 'get', raise_timeout)
    assert hu.check_health() is False


def test_generate_haiku_returns_error_when_unhealthy(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku returns error dict when health check fails."""
    monkeypatch.setattr(hu, 'check_health', lambda: False)
    result = hu.generate_haiku('тема')
    assert result['haiku_text'] == ''
    assert result['syllables_per_line'] == []
    assert result['total_words'] == 0
    assert result['error'] == 'Service not available'


def test_generate_haiku_success(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku returns json on success."""
    monkeypatch.setattr(hu, 'check_health', lambda: True)
    monkeypatch.setattr(
        hu.requests,
        'post',
        lambda _url, json, timeout: FakeResponse(
            {
                'haiku_text': 'Ветер шепчет мне\nО тайнах древних времен\nЛистья кружатся',
                'syllables_per_line': [5, 7, 5],
                'total_words': 9,
                'topic': 'осень',
            }
        ),
    )
    result = hu.generate_haiku('осень')
    assert result['haiku_text']
    assert result['syllables_per_line'] == [5, 7, 5]
    assert result['total_words'] == 9
    assert result['topic'] == 'осень'


def test_generate_haiku_connection_error(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku returns error dict on connection error."""
    monkeypatch.setattr(hu, 'check_health', lambda: True)

    def raise_connection_error(_url, json, timeout):
        raise hu.requests.exceptions.ConnectionError('Connection refused')

    monkeypatch.setattr(hu.requests, 'post', raise_connection_error)
    result = hu.generate_haiku('тема')
    assert result['error'] == 'Connection refused'
    assert result['haiku_text'] == ''


def test_generate_haiku_timeout(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku returns error dict on timeout."""
    monkeypatch.setattr(hu, 'check_health', lambda: True)

    def raise_timeout(_url, json, timeout):
        raise hu.requests.exceptions.Timeout('Timeout')

    monkeypatch.setattr(hu.requests, 'post', raise_timeout)
    result = hu.generate_haiku('тема')
    assert result['error'] == 'Timeout'
    assert result['haiku_text'] == ''


def test_generate_haiku_request_exception(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku returns error dict on request exception."""
    monkeypatch.setattr(hu, 'check_health', lambda: True)

    def raise_request_exception(_url, json, timeout):
        raise hu.requests.exceptions.RequestException('Request failed')

    monkeypatch.setattr(hu.requests, 'post', raise_request_exception)
    result = hu.generate_haiku('тема')
    assert result['error'] == 'Request failed'
    assert result['haiku_text'] == ''


def test_generate_haiku_generic_exception(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku returns error dict on generic exception."""
    monkeypatch.setattr(hu, 'check_health', lambda: True)

    def raise_generic_exception(_url, json, timeout):
        raise ValueError('Unexpected error')

    monkeypatch.setattr(hu.requests, 'post', raise_generic_exception)
    result = hu.generate_haiku('тема')
    assert result['error'] == 'Unexpected error'
    assert result['haiku_text'] == ''
