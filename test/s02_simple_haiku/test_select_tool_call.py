import json

import pytest

from src.s02_simple_haiku import select_tool_call as stc


def _mock_llm_response(message: dict) -> dict:
    return {'choices': [{'message': message}]}


@pytest.mark.unit
class TestSelectToolCallMockLLM:
    """Tests with mocked LLM API calls."""

    def test_select_tool_call_ok_with_args(self, monkeypatch):
        """Returns tool name and parsed arguments."""
        response = _mock_llm_response(
            {
                'function_call': {
                    'name': 'rag_search',
                    'arguments': json.dumps({'question': 'Что такое хайку?'}),
                }
            }
        )

        monkeypatch.setattr(
            stc,
            'post_chat_completions',
            lambda payload, verbose=False: response,
        )

        status, payload = stc.select_tool_call('Что такое хайку?')
        assert status == 0
        assert payload == ('rag_search', {'question': 'Что такое хайку?'})

    def test_select_tool_call_ok_without_args(self, monkeypatch):
        """Returns empty args when LLM omits arguments."""
        response = _mock_llm_response({'function_call': {'name': 'generate_haiku'}})

        monkeypatch.setattr(
            stc,
            'post_chat_completions',
            lambda payload, verbose=False: response,
        )

        status, payload = stc.select_tool_call('Напиши хайку')
        assert status == 0
        assert payload == ('generate_haiku', {})

    def test_select_tool_call_without_function_call(self, monkeypatch):
        """Returns None when function_call is missing."""
        response = _mock_llm_response({'content': 'Любой ответ'})

        monkeypatch.setattr(
            stc,
            'post_chat_completions',
            lambda payload, verbose=False: response,
        )

        status, payload = stc.select_tool_call('Любой запрос')
        assert status == 1
        assert payload is None

    def test_select_tool_call_llm_error(self, monkeypatch):
        """Returns None on LLM error response."""
        response = {'error': 'test error'}

        monkeypatch.setattr(
            stc,
            'post_chat_completions',
            lambda payload, verbose=False: response,
        )

        status, payload = stc.select_tool_call('Любой запрос')
        assert status == 2
        assert payload is None


@pytest.mark.llm
class TestSelectToolCallRealLLM:
    """Tests with real LLM API calls."""

    def test_select_tool_call_real_llm_rag(self):
        """Real LLM call, rag tool."""
        status, payload = stc.select_tool_call('Что такое хайку?', debug=True)
        assert status == 0
        tool_name, tool_args = payload
        assert tool_name == 'rag_search'
        assert isinstance(tool_args, dict)  # TODO

    def test_select_tool_call_real_llm_haiku(self):
        """Real LLM call, haiku tool."""
        status, payload = stc.select_tool_call('Напиши хайку', debug=True)
        assert status == 0
        tool_name, tool_args = payload
        assert tool_name == 'generate_haiku'
        assert isinstance(tool_args, dict)  # TODO
