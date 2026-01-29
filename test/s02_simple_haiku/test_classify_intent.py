"""Integration tests for classify_intent with real LLM calls."""

import pytest

from src.s02_simple_haiku.agent import classify_intent

freezing = 0.001


@pytest.mark.llm
class TestClassifyIntentIntegration:
    """Integration tests with real LLM API calls."""

    def test_ok_haiku(self):
        """Haiku generation requests."""
        cases = [
            'хайку про кота',
            'хокку мне о весне',
            'создай хайку',
            'пиши хаику',
            'сгенери хоку',
        ]
        for case in cases:
            result = classify_intent(case, temperature=freezing)
            assert result, f'Failed for: {case}'

    def test_ok_rag(self):
        """RAG requests."""
        cases = [
            'что такое хайку?',
            'объясни про хокку',
            'японская поэзия',
        ]
        for case in cases:
            result = classify_intent(case, temperature=freezing)
            assert result, f'Failed for: {case}'

    def test_deny_other(self):
        """Non-haiku and non-RAG requests."""
        cases = [
            'расскажи анекдот',
            'заказать выписку',
            'погода на завтра',
            'зеленый день когда',
            'топ рестиков на балитекущая процентная ставка?',
            'что лучше самокат или питбайк?',
            'whats the meaning of hive?',
        ]
        for case in cases:
            result = classify_intent(case, temperature=freezing)
            assert not result, f'Failed for: {case}'

    def test_edge_cases(self):
        """Edge cases."""
        # Very short request
        user_input = 'хайку'
        result = classify_intent(user_input, temperature=freezing)
        assert result, f'Failed for: {user_input}'

        # Mixed language
        user_input = 'write me a haiku'
        result = classify_intent(user_input, temperature=freezing)
        assert result, f'Failed for: {user_input}'

        # TODO: Ask for details
        user_input = 'напиши стишок'
        result = classify_intent(user_input, temperature=freezing)
        assert not result, f'Failed for: {user_input}'
