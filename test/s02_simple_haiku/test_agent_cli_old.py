import pytest

from src.s02_simple_haiku import agent

pytestmark = pytest.mark.skip


@pytest.mark.unit
def test_exit_command_empty_history_and_iteration(monkeypatch: pytest.MonkeyPatch):
    """
    EXIT_COMMANDS: first input is /exit -> message_history empty, iteration == 1.
    """
    inputs = iter(['/exit'])

    def fake_input(prompt: str) -> str:
        return next(inputs)

    monkeypatch.setattr('builtins.input', fake_input)
    result = agent.main()
    assert result['iteration'] == 1
    assert result['message_history'] == []


@pytest.mark.unit
def test_help_command_empty_history_and_iteration(monkeypatch: pytest.MonkeyPatch):
    """
    HELP_COMMANDS: /help then /exit -> message_history empty, iteration == 2.
    """
    inputs = iter(['/help', '/exit'])

    def fake_input(prompt: str) -> str:
        return next(inputs)

    monkeypatch.setattr('builtins.input', fake_input)
    result = agent.main()
    assert result['iteration'] == 2
    assert result['message_history'] == []


@pytest.mark.unit
def test_clear_command_clears_history(monkeypatch: pytest.MonkeyPatch):
    """
    CLEAR_COMMANDS: after adding messages, /clear empties message_history; exit returns empty history.
    """
    inputs = iter(['write a haiku', 'winter', '/clear', '/exit'])

    def fake_input(prompt: str) -> str:
        return next(inputs)

    monkeypatch.setattr('builtins.input', fake_input)
    result = agent.main()
    assert result['iteration'] == 4
    assert result['message_history'] == []
