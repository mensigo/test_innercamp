"""Smoke tests for agent integration with haiku and rag modules."""

import pytest

from src.s02_simple_haiku import agent

pytestmark = pytest.mark.unit


def test_display_haiku_with_stats():
    """Display haiku should format output with syllable stats."""
    haiku_text = 'Ветер шепчет мне\nО тайнах древних времен\nЛистья кружатся'
    stats = {
        'haiku_text': haiku_text,
        'syllables_per_line': [5, 7, 5],
        'total_words': 9,
        'topic': 'осень',
    }

    # Should not raise exception
    agent.display_haiku(haiku_text, stats)


def test_display_haiku_without_stats():
    """Display haiku should work without stats."""
    haiku_text = 'Ветер шепчет мне\nО тайнах древних времен\nЛистья кружатся'

    # Should not raise exception
    agent.display_haiku(haiku_text, None)


def test_display_haiku_empty():
    """Display haiku should handle empty haiku."""
    # Should not raise exception
    agent.display_haiku('', None)


def test_print_help():
    """Print help should not raise exception."""
    agent.print_help()


def test_print_reminder():
    """Print reminder should not raise exception."""
    agent.print_reminder()


def test_add_to_history():
    """Add message to history."""
    history = []
    agent.add_to_history(history, 'user', 'Привет')
    assert len(history) == 1
    assert history[0]['role'] == 'user'
    assert history[0]['content'] == 'Привет'


def test_add_to_history_limit():
    """Add messages to history should respect limit."""
    history = []
    # Добавляем больше сообщений, чем CONTEXT_HIST_LIMIT
    for i in range(15):
        agent.add_to_history(history, 'user', f'Сообщение {i}')

    assert len(history) == agent.CONTEXT_HIST_LIMIT
    # Проверяем, что остались последние сообщения
    assert history[0]['content'] == 'Сообщение 5'
    assert history[-1]['content'] == 'Сообщение 14'
