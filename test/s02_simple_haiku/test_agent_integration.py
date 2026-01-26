"""Integration tests for classify_intent with real LLM calls."""

import pytest

from src.s01_simple_haiku.agent import classify_intent

freezing = 0.001


@pytest.mark.integration
class TestClassifyIntentIntegration:
    """Integration tests with real LLM API calls."""

    def test_haiku_request_with_topic_real_llm(self):
        """Test real LLM call with haiku request including topic."""
        result = classify_intent('напиши хайку о море', temperature=freezing)
        assert result is True

    def test_haiku_request_without_topic_real_llm(self):
        """Test real LLM call with haiku request without topic."""
        result = classify_intent('сгенерируй хокку', temperature=freezing)
        assert result is True

    def test_haiku_variations_real_llm(self):
        """Test real LLM call with various haiku requests."""
        test_cases = [
            ('хайку про кота', True),
            ('хоку мне о весне', True),
            ('создай хайку', True),
        ]

        for user_input, expected in test_cases:
            result = classify_intent(user_input, temperature=freezing)
            assert result == expected, f'Failed for: {user_input}'

    def test_non_haiku_requests_real_llm(self):
        """Test real LLM call with non-haiku requests."""
        test_cases = [
            ('напиши стишок', False),
            ('расскажи анекдот', False),
            ('что такое хайку?', False),
            ('объясни про хокку', False),
        ]

        for user_input, expected in test_cases:
            result = classify_intent(user_input, temperature=freezing)
            assert result == expected, f'Failed for: {user_input}'

    def test_edge_cases_real_llm(self):
        """Test real LLM call with edge cases."""
        # Very short request
        user_input = 'хайку'
        result = classify_intent(user_input, temperature=freezing)
        assert result is True, f'Failed for: {user_input}'

        # Mixed language
        user_input = 'write me a haiku'
        result = classify_intent(user_input, temperature=freezing)
        assert result is True, f'Failed for: {user_input}'
