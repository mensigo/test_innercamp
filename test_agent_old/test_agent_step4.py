"""Step4-execute tests.

Classify/select calls go to the real LLM. We monkeypatch `requests.get`/`requests.post`
only when the URL targets the local haiku service, to simulate health/generate failures.
All other HTTP calls are untouched.
"""

import json as jsonlib

import pytest
import requests

from src_agent.agent import agent
from src_agent.config import config

pytestmark = pytest.mark.step4


@pytest.mark.llm
def test_agent_execute_haiku_health_fail(monkeypatch: pytest.MonkeyPatch):
    """
    If haiku service health fails, execute should return 401 and stop.
    """
    real_get = requests.get
    health_hits = {'count': 0}

    def fake_get(url: str, **kwargs):
        if url == f'http://localhost:{config.tool_haiku_port}/health':
            health_hits['count'] += 1

            class DummyResponse:
                def __init__(self, payload: dict):
                    self._payload = payload
                    self.text = jsonlib.dumps(payload, ensure_ascii=False)
                    self.status_code = 200

                def raise_for_status(self):
                    return None

                def json(self) -> dict:
                    return self._payload

                def __repr__(self) -> str:
                    return f'DummyResponse(status_code={self.status_code})'

            return DummyResponse({'status': 'fail'})

        return real_get(url, **kwargs)

    monkeypatch.setattr('requests.get', fake_get)

    message_history = [{'role': 'user', 'content': 'напиши хайку про осень'}]

    result = agent(message_history)

    assert health_hits['count'] >= 1, 'health check not called'
    assert result.get('last_state') == 'execute'
    assert result.get('classify', {}).get('code') == 100
    assert result.get('select', {}).get('code') == 200
    assert result.get('validate', {}).get('code') == 300

    execute = result.get('execute', {})
    assert execute.get('code') == 401
    assert (
        execute.get('message')
        == 'Небольшие трудности при генерации хайку.. Инженеры уже работают над запуском сервиса..'
    )


@pytest.mark.llm
def test_agent_execute_haiku_post_error(monkeypatch: pytest.MonkeyPatch):
    """
    If haiku generate endpoint fails, execute should return 401 with generic error.
    """

    real_get = requests.get
    real_post = requests.post
    health_hits = {'count': 0}
    post_hits = {'count': 0}

    def fake_get(url: str, **kwargs):
        if url == f'http://localhost:{config.tool_haiku_port}/health':
            health_hits['count'] += 1

            class DummyResponse:
                def __init__(self, payload: dict):
                    self._payload = payload
                    self.text = jsonlib.dumps(payload, ensure_ascii=False)
                    self.status_code = 200

                def raise_for_status(self):
                    return None

                def json(self) -> dict:
                    return self._payload

                def __repr__(self) -> str:
                    return f'DummyResponse(status_code={self.status_code})'

            return DummyResponse({'status': 'ok'})

        return real_get(url, **kwargs)

    def fake_post(url: str, json: dict, **kwargs):
        if url == f'http://localhost:{config.tool_haiku_port}/generate_haiku':
            post_hits['count'] += 1
            raise requests.exceptions.Timeout('mock timeout to haiku service')
        return real_post(url, json=json, **kwargs)

    monkeypatch.setattr('requests.get', fake_get)
    monkeypatch.setattr('requests.post', fake_post)

    message_history = [{'role': 'user', 'content': 'напиши хайку про осень'}]

    result = agent(message_history)

    assert health_hits['count'] >= 1, 'health check not called'
    assert post_hits['count'] >= 1, 'generate endpoint not called'

    assert result.get('last_state') == 'execute'
    assert result.get('classify', {}).get('code') == 100
    assert result.get('select', {}).get('code') == 200
    assert result.get('validate', {}).get('code') == 300

    execute = result.get('execute', {})
    assert execute.get('code') == 401
    assert (
        execute.get('message')
        == 'Произошла чудовищная ошибка при генерации хайку.. Тысяча извинений! Попробуем снова?'
    )
