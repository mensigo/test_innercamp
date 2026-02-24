"""Smoke tests for agent integration with haiku and rag modules."""

import pytest

from src.s02_simple_haiku import agent

pytestmark = pytest.mark.llm


@pytest.mark.unit
@pytest.mark.parametrize('limit', [3, 7])
def test_message_history_trimmed_by_context_limit(
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
):
    """
    With custom CONTEXT_HIST_LIMIT, classify mocked to return 101 (irrelevant),
    stack user+assistant messages and assert history is cut to at most limit.
    """
    monkeypatch.setattr(agent, 'CONTEXT_HIST_LIMIT', limit)

    def fake_classify(message_history: list, **kwargs) -> int:
        return 101  # irrelevant -> only add user + assistant, no select/execute

    monkeypatch.setattr(
        'src.s02_simple_haiku.agent.classify_intent',
        fake_classify,
    )

    # Enough inputs to exceed limit: each adds user + assistant (2 messages)
    num_turns = 5
    input_list = [f'query_{i}' for i in range(num_turns)] + ['/exit']
    inputs = iter(input_list)

    def fake_input(prompt: str) -> str:
        return next(inputs)

    monkeypatch.setattr('builtins.input', fake_input)
    result = agent.main()

    message_history = result['message_history']
    assert len(message_history) == limit
    assert result['iteration'] == num_turns + 1


@pytest.mark.llm
def test_main_smoke_write_haiku_e2e(monkeypatch: pytest.MonkeyPatch):
    """
    Smoke e2e: main() runs with 'write a haiku' input and exits without error.
    Only input() is mocked; classify_intent and select_tool_call use real LLM.
    """
    inputs = iter(['write a haiku', 'winter', '/exit'])

    def fake_input(prompt: str) -> str:
        return next(inputs)

    monkeypatch.setattr('builtins.input', fake_input)
    result = agent.main()
    assert result['iteration'] == 3
    message_history = result['message_history']
    assert len(message_history) > 0
    assert message_history[-1]['role'] == 'assistant'
    assert 'Хайку:' in message_history[-1]['content']


@pytest.mark.llm
def test_chain_haiku_russian_multi_turn_1(monkeypatch: pytest.MonkeyPatch):
    """
    E2e chain: напиши хайку -> кокосы в меду -> еще раз -> еще -> /exit.
    Completes without error; history has multiple user/assistant turns; at least one haiku reply.
    """
    inputs = iter(
        [
            'напиши хайку',
            'кокосы в меду',
            'еще раз',
            'еще',
            '/exit',
        ]
    )

    def fake_input(prompt: str) -> str:
        return next(inputs)

    monkeypatch.setattr('builtins.input', fake_input)
    result = agent.main()
    assert result['iteration'] == 5
    message_history = result['message_history']
    assert len(message_history) > 0
    assistant_haiku_messages = [
        m
        for m in message_history
        if m.get('role') == 'assistant' and 'Хайку:' in m.get('content', '')
    ]
    assert len(assistant_haiku_messages) >= 1
