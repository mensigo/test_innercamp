"""Step2-select stage tests: classify hits real LLM, select calls are intercepted.

We only monkeypatch requests.post when payload contains tool/function keys. All other
calls (classify, etc.) flow to the real requests.post. Intercepted calls capture the
payload for assertions and return crafted responses to drive select branches (201/202/203).
"""

import pytest
import requests

from src_agent.agent import agent

pytestmark = pytest.mark.step2


@pytest.mark.llm
def test_agent_select_no_tool_stops_flow(monkeypatch: pytest.MonkeyPatch):
    """
    When step2:select returns 201 (no tool), pipeline should stop after select.
    """

    captured = []
    real_post = requests.post

    def fake_post(url: str, json: dict, **kwargs):
        payload_keys = set(json.keys())
        has_tool_schema = any(
            key in payload_keys
            for key in ('function', 'functions', 'tools', 'tool_choice')
        )

        if has_tool_schema:
            captured.append(json)

            class DummyResponse:
                def __init__(self, payload: dict):
                    self._payload = payload
                    self.text = repr(payload)
                    self.status_code = 200

                def __repr__(self) -> str:
                    return f'DummyResponse(status_code={self.status_code}, payload={self._payload})'

                def raise_for_status(self):
                    return None

                def json(self) -> dict:
                    return self._payload

            # select -> return message without tool_call/function_call => 201
            return DummyResponse({'choices': [{'message': {}}]})

        # otherwise let real classify call proceed
        return real_post(url, json=json, **kwargs)

    monkeypatch.setattr('requests.post', fake_post)

    message_history = [
        {'role': 'user', 'content': 'напиши хайку'},
    ]

    result = agent(message_history)

    assert any(
        any(
            k in (payload or {})
            for k in ('function', 'functions', 'tools', 'tool_choice')
        )
        for payload in captured
    ), 'no function call request detected'

    assert result.get('last_state') == 'select'
    assert result.get('classify', {}).get('code') == 100  # relevant
    assert result.get('select', {}).get('code') == 201
    assert (
        result.get('select', {}).get('message')
        == 'Не удалось определить инструмент. Просьба переформулировать запрос.'
    )


@pytest.mark.llm
def test_agent_select_parse_error_stops_flow(monkeypatch: pytest.MonkeyPatch):
    """
    When step2:select cannot parse tool call (202), pipeline should stop after select.
    """

    captured = []
    real_post = requests.post

    def fake_post(url: str, json: dict, **kwargs):
        payload_keys = set(json.keys())
        has_tool_schema = any(
            key in payload_keys
            for key in ('function', 'functions', 'tools', 'tool_choice')
        )

        if has_tool_schema:
            captured.append(json)

            class DummyResponse:
                def __init__(self, payload: dict):
                    self._payload = payload
                    self.text = repr(payload)
                    self.status_code = 200

                def __repr__(self) -> str:
                    return f'DummyResponse(status_code={self.status_code}, payload={self._payload})'

                def raise_for_status(self):
                    return None

                def json(self) -> dict:
                    return self._payload

            # select -> kinda malformed JSON to trigger 202
            return DummyResponse(
                {
                    'choices': [
                        {
                            'message': {
                                'function_call': {
                                    'name': 'generate_haiku',
                                    'arguments': '{"theme": "Бурное море", BAD_JSON}',
                                },
                                'tool_calls': [
                                    {
                                        'function': {
                                            'name': 'generate_haiku',
                                            'arguments': '{"theme": "Бурное море", BAD_JSON}',
                                        }
                                    }
                                ],
                            }
                        }
                    ]
                }
            )

        # otherwise let real classify call proceed
        return real_post(url, json=json, **kwargs)

    monkeypatch.setattr('requests.post', fake_post)

    message_history = [
        {'role': 'user', 'content': 'напиши хайку'},
    ]

    result = agent(message_history)

    assert any(
        any(
            k in (payload or {})
            for k in ('function', 'functions', 'tools', 'tool_choice')
        )
        for payload in captured
    ), 'no function call request detected'

    assert result.get('last_state') == 'select'
    assert result.get('classify', {}).get('code') == 100  # relevant
    assert result.get('select', {}).get('code') == 202
    assert (
        result.get('select', {}).get('message')
        == 'Ошибка при разборе ответа LLM, завершаюсь..'
    )


@pytest.mark.llm
def test_agent_select_request_error_branch(monkeypatch: pytest.MonkeyPatch):
    """
    When step2:select gets an LLM request error (203), pipeline stops after select.
    """

    captured = []
    real_post = requests.post

    def fake_post(url: str, json: dict, **kwargs):
        payload_keys = set(json.keys())
        has_tool_schema = any(
            key in payload_keys
            for key in ('function', 'functions', 'tools', 'tool_choice')
        )

        if has_tool_schema:
            captured.append(json)

            class DummyResponse:
                def __init__(self, payload: dict):
                    self._payload = payload
                    self.text = repr(payload)
                    self.status_code = 200

                def __repr__(self) -> str:
                    return f'DummyResponse(status_code={self.status_code}, payload={self._payload})'

                def raise_for_status(self):
                    return None

                def json(self) -> dict:
                    return self._payload

            # select -> respond with error to trigger 203
            return DummyResponse({'error': 'mock error', 'choices': [{'message': {}}]})

        # otherwise let real classify call proceed
        return real_post(url, json=json, **kwargs)

    monkeypatch.setattr('requests.post', fake_post)

    message_history = [
        {'role': 'user', 'content': 'напиши хайку'},
    ]

    result = agent(message_history)

    assert any(
        any(
            k in (payload or {})
            for k in ('function', 'functions', 'tools', 'tool_choice')
        )
        for payload in captured
    ), 'no function call request detected'

    assert result.get('last_state') == 'select'
    assert result.get('classify', {}).get('code') == 100  # relevant
    assert result.get('select', {}).get('code') == 203
    assert (
        result.get('select', {}).get('message')
        == 'Ошибка при запросе LLM, завершаюсь..'
    )
