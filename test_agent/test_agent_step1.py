"""Step1-classify tests: we stub requests.post only for classify to drive 102/103 branches.

Tests verify classify codes/messages and that the pipeline halts before select. Non-stubbed
paths would hit real LLM, but here we replace requests.post to emulate parse and request errors.
"""

import pytest
import requests

from src_agent.agent import agent

pytestmark = pytest.mark.step1


@pytest.mark.llm
@pytest.mark.parametrize(
    'user_text',
    [
        'расскажи анекдот',
        'заказать выписку',
        'погода на завтра',
        'зеленый день когда',
        'топ рестиков на бали',
        'текущая процентная ставка?',
        'что лучше самокат или питбайк?',
        'whats the meaning of hive?',
    ],
)
def test_agent_classify_non_relevant_stops_flow(user_text: str):
    """For irrelevant queries classify must return 101 and stop pipeline."""
    message_history = [{'role': 'user', 'content': user_text}]

    result = agent(message_history)

    assert result.get('last_state') == 'classify'
    assert result.get('classify', {}).get('code') == 101
    assert (
        result.get('classify', {}).get('message', '')
        == 'Запрос не связан с функционалом агента.'
    )

    assert result.get('select') == {}  # went no further


@pytest.mark.unit
def test_agent_classify_parse_error_stops_flow(monkeypatch: pytest.MonkeyPatch):
    """Malformed LLM payload triggers parse error (code 102) and stops."""

    def fake_request_post(url: str, json: dict, **kwargs):
        class DummyResponse:
            def __init__(self):
                self.text = ''
                self.status_code = 200

            def raise_for_status(self):
                return None

            def json(self) -> dict:
                # Missing required fields -> classify_intent parse error (102)
                return {}

        return DummyResponse()

    monkeypatch.setattr('requests.post', fake_request_post)

    message_history = [{'role': 'user', 'content': 'любая строка'}]

    result = agent(message_history)

    assert result.get('last_state') == 'classify'
    assert result.get('classify', {}).get('code') == 102
    assert (
        result.get('classify', {}).get('message')
        == 'Ошибка при разборе ответа LLM, завершаюсь..'
    )
    assert result.get('select') == {}  # pipeline halted after classify


@pytest.mark.unit
def test_agent_classify_request_error_stops_flow(monkeypatch: pytest.MonkeyPatch):
    """Network/HTTP issues return code 103 and stop pipeline."""

    def fake_post(url: str, json: dict, **kwargs):
        raise requests.exceptions.Timeout('mock timeout')

    monkeypatch.setattr('requests.post', fake_post)

    message_history = [{'role': 'user', 'content': 'another query'}]

    result = agent(message_history)

    assert result.get('last_state') == 'classify'
    assert result.get('classify', {}).get('code') == 103
    assert (
        result.get('classify', {}).get('message')
        == 'Ошибка при запросе LLM, завершаюсь..'
    )
    assert result.get('select') == {}  # pipeline halted after classify
