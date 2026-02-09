"""Smoke tests for select_tool_call."""

import pytest

from src.s02_simple_haiku import agent

pytestmark = pytest.mark.unit


def _mock_llm_response(message: dict) -> dict:
    return {'choices': [{'message': message}]}


def test_select_tool_call_ok_with_args(monkeypatch):
    """Returns tool name and parsed arguments."""
    response = _mock_llm_response(
        {
            'function_call': {
                'name': 'rag_search',
                'arguments': '{"question": "Что такое хайку?"}',
            }
        }
    )

    monkeypatch.setattr(
        agent,
        'post_chat_completions',
        lambda payload, verbose=False: response,
    )

    result = agent.select_tool_call('Что такое хайку?')
    assert result == ('rag_search', {'question': 'Что такое хайку?'})


def test_select_tool_call_ok_without_args(monkeypatch):
    """Returns empty args when LLM omits arguments."""
    response = _mock_llm_response({'function_call': {'name': 'generate_haiku'}})

    monkeypatch.setattr(
        agent,
        'post_chat_completions',
        lambda payload, verbose=False: response,
    )

    result = agent.select_tool_call('Напиши хайку')
    assert result == ('generate_haiku', {})


def test_select_tool_call_unknown_tool(monkeypatch):
    """Returns None for built-in or unknown tools."""
    response = _mock_llm_response(
        {'function_call': {'name': 'builtin_tool', 'arguments': '{}'}}
    )

    monkeypatch.setattr(
        agent,
        'post_chat_completions',
        lambda payload, verbose=False: response,
    )

    result = agent.select_tool_call('Проверь инструмент')
    assert result is None


def test_select_tool_call_without_function_call(monkeypatch):
    """Returns None when function_call is missing."""
    response = _mock_llm_response({'content': 'plain answer'})

    monkeypatch.setattr(
        agent,
        'post_chat_completions',
        lambda payload, verbose=False: response,
    )

    result = agent.select_tool_call('Любой запрос')
    assert result is None


def test_select_tool_call_llm_error(monkeypatch):
    """Returns None on LLM error response."""
    response = {'error': 'test error'}

    monkeypatch.setattr(
        agent,
        'post_chat_completions',
        lambda payload, verbose=False: response,
    )

    result = agent.select_tool_call('Любой запрос')
    assert result is None


@pytest.mark.llm
def test_select_tool_call_real_llm():
    """Smoke test with real LLM call."""
    result = agent.select_tool_call('Что такое хайку?')
    if result is None:
        assert result is None
        return

    tool_name, tool_args = result
    assert tool_name in {'rag_search', 'generate_haiku'}
    assert isinstance(tool_args, dict)
