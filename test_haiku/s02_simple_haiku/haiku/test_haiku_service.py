"""Smoke tests for haiku_service endpoints."""

import importlib
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.haiku]


def load_haiku_service(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """
    Load haiku_service with patched post_chat_completions.
    """
    module = importlib.import_module('src.s02_simple_haiku.haiku.haiku_service')
    return importlib.reload(module)


def test_health_ok(monkeypatch: pytest.MonkeyPatch):
    """Health should always return ok for haiku service."""
    haiku_service = load_haiku_service(monkeypatch)
    client = haiku_service.app.test_client()

    response = client.get('/health')
    data = response.get_json()

    assert response.status_code == 200
    assert data['status'] == 'ok'


def test_generate_haiku_success(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku should return haiku text with stats."""
    haiku_service = load_haiku_service(monkeypatch)

    def stub_post_chat_completions(_payload, _verbose=False):
        return {
            'choices': [
                {
                    'message': {
                        'content': 'Ветер шепчет мне\nО тайнах древних времен\nЛистья кружатся'
                    }
                }
            ]
        }

    monkeypatch.setattr(
        haiku_service, 'post_chat_completions', stub_post_chat_completions
    )

    client = haiku_service.app.test_client()
    response = client.post('/generate_haiku', json={'theme': 'осень'})
    data = response.get_json()

    assert response.status_code == 200
    assert 'haiku_text' in data
    assert 'syllables_per_line' in data
    assert 'total_words' in data
    assert data['theme'] == 'осень'
    assert len(data['syllables_per_line']) == 3
    assert data['total_words'] > 0


def test_generate_haiku_missing_theme(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku should return 400 without theme."""
    haiku_service = load_haiku_service(monkeypatch)
    client = haiku_service.app.test_client()

    response = client.post('/generate_haiku', json={})
    data = response.get_json()

    assert response.status_code == 400
    assert 'error' in data
    assert data['error'] == 'Missing theme'


def test_generate_haiku_llm_error(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku should return 500 when LLM fails."""
    haiku_service = load_haiku_service(monkeypatch)

    def stub_post_chat_completions(_payload, _verbose=False):
        return {'error': 'LLM unavailable'}

    monkeypatch.setattr(
        haiku_service, 'post_chat_completions', stub_post_chat_completions
    )

    client = haiku_service.app.test_client()
    response = client.post('/generate_haiku', json={'theme': 'зима'})
    data = response.get_json()

    assert response.status_code == 500
    assert 'error' in data
    assert data['error'] == 'Generation failed'


def test_generate_haiku_internal_error(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku should return 500 on unexpected errors."""
    haiku_service = load_haiku_service(monkeypatch)

    def stub_post_chat_completions(_payload, _verbose=False):
        return {
            'choices': [
                {
                    'message': {
                        'content': 'Ветер шепчет мне\nМоре в тишине\nЗакат золотой'
                    }
                }
            ]
        }

    def boom(_text):
        raise RuntimeError('boom')

    monkeypatch.setattr(
        haiku_service, 'post_chat_completions', stub_post_chat_completions
    )
    monkeypatch.setattr(haiku_service, 'count_syllables_and_words', boom)

    client = haiku_service.app.test_client()
    response = client.post('/generate_haiku', json={'theme': 'осень'})
    data = response.get_json()

    assert response.status_code == 500
    assert data['error'] == 'Internal server error'


def test_generate_haiku_function(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku function should call LLM and return text."""
    from src.s02_simple_haiku.haiku import haiku_service

    def stub_post_chat_completions(_payload, _verbose=False):
        return {
            'choices': [
                {
                    'message': {
                        'content': 'Тест хайку текст\nВторая строка здесь\nТретья строка тут'
                    }
                }
            ]
        }

    monkeypatch.setattr(
        haiku_service, 'post_chat_completions', stub_post_chat_completions
    )

    result = haiku_service.generate_haiku('тест')

    assert result == 'Тест хайку текст\nВторая строка здесь\nТретья строка тут'


def test_generate_haiku_function_with_temperature(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku function should accept temperature parameter."""
    from src.s02_simple_haiku.haiku import haiku_service

    captured_payload = {}

    def capture_payload(payload, verbose=False):
        captured_payload.update(payload)
        return {'choices': [{'message': {'content': 'Хайку текст'}}]}

    monkeypatch.setattr(haiku_service, 'post_chat_completions', capture_payload)

    haiku_service.generate_haiku('тест', temperature=0.8)

    assert captured_payload['temperature'] == 0.8


def test_generate_haiku_function_llm_error(monkeypatch: pytest.MonkeyPatch):
    """Generate haiku function should return empty string on LLM error."""
    from src.s02_simple_haiku.haiku import haiku_service

    def stub_post_chat_completions(_payload, _verbose=False):
        return {'error': 'fail'}

    monkeypatch.setattr(
        haiku_service, 'post_chat_completions', stub_post_chat_completions
    )

    assert haiku_service.generate_haiku('тема') == ''
