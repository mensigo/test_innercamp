"""Step3-validate stage tests: classify/select hit real LLM, validate fails.

Tool-call requests are captured by monkeypatching requests.post only when the
payload includes tool/function keys; all other calls go to the real LLM.
"""

import json as jsonlib

import pytest
import requests

from src_agent.agent import agent
from src_agent.config import config

pytestmark = pytest.mark.step3


@pytest.mark.llm
def test_agent_validate_theme_too_long(monkeypatch: pytest.MonkeyPatch):
    """
    When select returns generate_haiku with an overly long theme, validate must
    stop the pipeline and expose reason/param in the result.
    """

    long_theme = 'очень замысловатая и невероятно длинная тема для хайку'
    if len(long_theme) <= config.tool_haiku_max_theme_len:
        long_theme = 'X' * (config.tool_haiku_max_theme_len + 5)

    captured = []
    real_post = requests.post

    def fake_post(url: str, json: dict, **kwargs):
        payload = json
        payload_keys = set(payload.keys())
        has_tool_schema = any(
            key in payload_keys
            for key in ('function', 'functions', 'tools', 'tool_choice')
        )

        if has_tool_schema:
            captured.append(payload)

            class DummyResponse:
                def __init__(self, payload: dict):
                    self._payload = payload
                    self.text = repr(payload)
                    self.status_code = 200

                def raise_for_status(self):
                    return None

                def json(self) -> dict:
                    return self._payload

            return DummyResponse(
                {
                    'choices': [
                        {
                            'message': {
                                'function_call': {
                                    'name': 'generate_haiku',
                                    'arguments': jsonlib.dumps(
                                        {'theme': long_theme}, ensure_ascii=False
                                    ),
                                },
                                'tool_calls': [
                                    {
                                        'type': 'function',
                                        'function': {
                                            'name': 'generate_haiku',
                                            'arguments': jsonlib.dumps(
                                                {'theme': long_theme},
                                                ensure_ascii=False,
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )

        return real_post(url, json=payload, **kwargs)

    monkeypatch.setattr('requests.post', fake_post)

    message_history = [
        {'role': 'user', 'content': f'напиши хайку про {long_theme}'},
    ]

    result = agent(message_history)

    assert captured, 'select step did not trigger a tool call request'

    assert result.get('last_state') == 'validate'
    assert result.get('classify', {}).get('code') == 100  # relevant
    assert result.get('select', {}).get('code') == 200  # tool selected

    validate = result.get('validate', {})
    assert validate.get('code') == 0
    assert validate.get('param') == 'theme'
    assert validate.get('reason') == 'long'
    assert (
        validate.get('message')
        == 'Тема слишком длинная. Просьба сформулировать более кратко.'
    )

    assert result.get('execute') == {}  # pipeline halted before execute


@pytest.mark.llm
def test_agent_validate_theme_missing(monkeypatch: pytest.MonkeyPatch):
    """Missing theme param should fail validation with reason=missing."""

    captured = []
    real_post = requests.post

    def fake_post(url: str, json: dict, **kwargs):
        payload = json
        payload_keys = set(payload.keys())
        has_tool_schema = any(
            key in payload_keys
            for key in ('function', 'functions', 'tools', 'tool_choice')
        )

        if has_tool_schema:
            captured.append(payload)

            class DummyResponse:
                def __init__(self, payload: dict):
                    self._payload = payload
                    self.text = repr(payload)
                    self.status_code = 200

                def raise_for_status(self):
                    return None

                def json(self) -> dict:
                    return self._payload

            return DummyResponse(
                {
                    'choices': [
                        {
                            'message': {
                                'function_call': {
                                    'name': 'generate_haiku',
                                    'arguments': jsonlib.dumps({}, ensure_ascii=False),
                                },
                                'tool_calls': [
                                    {
                                        'type': 'function',
                                        'function': {
                                            'name': 'generate_haiku',
                                            'arguments': jsonlib.dumps(
                                                {}, ensure_ascii=False
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )

        return real_post(url, json=payload, **kwargs)

    monkeypatch.setattr('requests.post', fake_post)

    message_history = [{'role': 'user', 'content': 'напиши хайку'}]

    result = agent(message_history)

    assert captured, 'select step did not trigger a tool call request'
    assert result.get('last_state') == 'validate'
    assert result.get('classify', {}).get('code') == 100
    assert result.get('select', {}).get('code') == 200

    validate = result.get('validate', {})
    assert validate.get('code') == 0
    assert validate.get('param') == 'theme'
    assert validate.get('reason') == 'missing'
    assert validate.get('message') == 'Не совсем понял тему хайку. Какую использовать?'
    assert result.get('execute') == {}


@pytest.mark.llm
def test_agent_validate_theme_empty(monkeypatch: pytest.MonkeyPatch):
    """Blank theme value should fail validation with reason=empty."""

    captured = []
    real_post = requests.post

    def fake_post(url: str, json: dict, **kwargs):
        payload = json
        payload_keys = set(payload.keys())
        has_tool_schema = any(
            key in payload_keys
            for key in ('function', 'functions', 'tools', 'tool_choice')
        )

        if has_tool_schema:
            captured.append(payload)

            class DummyResponse:
                def __init__(self, payload: dict):
                    self._payload = payload
                    self.text = repr(payload)
                    self.status_code = 200

                def raise_for_status(self):
                    return None

                def json(self) -> dict:
                    return self._payload

            return DummyResponse(
                {
                    'choices': [
                        {
                            'message': {
                                'function_call': {
                                    'name': 'generate_haiku',
                                    'arguments': jsonlib.dumps(
                                        {'theme': '   '}, ensure_ascii=False
                                    ),
                                },
                                'tool_calls': [
                                    {
                                        'type': 'function',
                                        'function': {
                                            'name': 'generate_haiku',
                                            'arguments': jsonlib.dumps(
                                                {'theme': ''}, ensure_ascii=False
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )

        return real_post(url, json=payload, **kwargs)

    monkeypatch.setattr('requests.post', fake_post)

    message_history = [{'role': 'user', 'content': 'напиши хайку'}]

    result = agent(message_history)

    assert captured, 'select step did not trigger a tool call request'
    assert result.get('last_state') == 'validate'
    assert result.get('classify', {}).get('code') == 100
    assert result.get('select', {}).get('code') == 200

    validate = result.get('validate', {})
    assert validate.get('code') == 0
    assert validate.get('param') == 'theme'
    assert validate.get('reason') == 'empty'
    assert validate.get('message') == 'Не совсем понял тему хайку. Какую использовать?'
    assert result.get('execute') == {}
