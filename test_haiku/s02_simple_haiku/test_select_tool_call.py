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
        assert status == 200
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
        assert status == 200
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
        assert status == 201
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
        assert status == 203
        assert payload is None


@pytest.mark.llm
class TestSelectToolCallRealLLM:
    """Tests with real LLM API calls."""

    @pytest.mark.parametrize(
        'user_input',
        [
            'Что такое хайку?',
            'Когда был написан манъесю',
            'Есть ли в старояпонском священные цифры?',
        ],
    )
    def test_select_tool_call_real_llm_rag_single(self, user_input: str):
        """
        Real LLM call, one-string input, rag_search tool.
        """
        status, payload = stc.select_tool_call(user_input, verbose=True)
        assert status == 200
        tool_name, tool_args = payload
        assert tool_name == 'rag_search'
        assert isinstance(tool_args, dict)
        assert 'question' in tool_args

    @pytest.mark.parametrize(
        'history',
        [
            [
                {'role': 'user', 'content': 'Расскажи про хокку'},
                {'role': 'assistant', 'content': 'Хокку - это...'},
                {'role': 'user', 'content': 'Откуда появилось хокку?'},
            ],
            [
                {'role': 'user', 'content': 'Что такое манъёсю?'},
                {'role': 'assistant', 'content': 'Манъёсю - это...'},
                {'role': 'user', 'content': 'Почему оно важно?'},
            ],
            [
                {'role': 'user', 'content': 'Расскажи о японской поэзии'},
                {'role': 'assistant', 'content': 'Японская поэзия включает...'},
                {'role': 'user', 'content': 'Какие там есть жанры?'},
            ],
        ],
    )
    def test_select_tool_call_real_llm_rag_history(self, history: list[dict]):
        """
        Real LLM call, multi-message history, rag_search tool.
        """
        status, payload = stc.select_tool_call(history, verbose=True)
        assert status == 200
        tool_name, tool_args = payload
        assert tool_name == 'rag_search'
        assert isinstance(tool_args, dict)
        assert 'question' in tool_args

    @pytest.mark.parametrize(
        'user_input',
        [
            'Напиши хайку',
            'Сделай хокку',
            'Хайку',
            'Пиши хокку мне',
        ],
    )
    def test_select_tool_call_real_llm_haiku_no_theme(self, user_input: str):
        """Real LLM call, haiku tool, one-string input, no explicit theme."""
        status, payload = stc.select_tool_call(user_input, verbose=True)
        assert status == 200
        tool_name, tool_args = payload
        assert tool_name == 'generate_haiku'
        assert isinstance(tool_args, dict)
        assert 'theme' not in tool_args or not tool_args['theme']

    @pytest.mark.parametrize(
        'user_input,expected_theme',
        [
            ('Сгенерируй хайку на тему красота', 'красота'),
            ('Создай хайку о первом снеге', 'первый снег'),
            ('Хокку про дождь', 'дождь'),
            ('Напиши хокку на тему крылатые рыбкиы', 'крылатые рыбки'),
        ],
    )
    def test_select_tool_call_real_llm_haiku_with_theme(
        self, user_input: str, expected_theme: str
    ):
        """Real LLM call, haiku tool, one-string input, explicit theme."""
        status, payload = stc.select_tool_call(user_input, verbose=True)
        assert status == 200
        tool_name, tool_args = payload
        assert tool_name == 'generate_haiku'
        assert isinstance(tool_args, dict)
        assert 'theme' in tool_args
        assert isinstance(tool_args['theme'], str)
        assert expected_theme.lower() in tool_args['theme'].lower()
