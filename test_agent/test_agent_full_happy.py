"""End-to-end happy path test without mocks."""

import pytest

from src_agent.agent import agent


@pytest.mark.llm
@pytest.mark.happy
def test_agent_full_happy_path():
    """
    Full happy flow: classify -> select -> validate -> execute (generate_haiku).
    """

    message_history = [{'role': 'user', 'content': 'хайку про лето'}]

    result = agent(message_history)

    assert result.get('last_state') == 'execute'
    assert result.get('execute', {}).get('code') == 400

    execute = result.get('execute', {})
    assert execute.get('message')
    assert execute.get('syllables_msg')
    assert execute.get('total_words')
