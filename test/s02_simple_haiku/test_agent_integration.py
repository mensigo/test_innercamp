"""Smoke tests for agent integration with haiku and rag modules."""

import pytest

from src.s02_simple_haiku import agent

pytestmark = pytest.mark.llm


def test_main_smoke_write_haiku_e2e(monkeypatch: pytest.MonkeyPatch):
    """
    Smoke e2e: main() runs with 'write a haiku' input and exits without error.
    Only input() is mocked; classify_intent and select_tool_call use real LLM.
    """
    inputs = iter(['write a haiku', 'winter', '/exit'])

    def fake_input(prompt: str) -> str:
        return next(inputs)

    monkeypatch.setattr('builtins.input', fake_input)
    message_history = agent.main()
    assert len(message_history) > 0
    assert message_history[-1]['role'] == 'assistant'
    assert 'Хайку:' in message_history[-1]['content']
