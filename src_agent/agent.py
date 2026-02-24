"""CLI agent with intent classification and tool routing."""

from src.s02_simple_haiku.agent import agent as agent_impl
from src.s02_simple_haiku.agent import main as main_impl


def agent(message_history: list[dict]) -> dict:
    """
    Run classify/select/validate/execute pipeline for current history.
    """
    return agent_impl(message_history)


def main() -> dict:
    """
    (optional) Main interactive loop for the haiku agent.
    """
    return main_impl()


if __name__ == '__main__':
    main()
