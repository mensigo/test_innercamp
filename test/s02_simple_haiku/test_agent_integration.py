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


def test_validate_tool_call_rag_search():
    """Validate rag_search tool call."""
    result = agent.validate_tool_call(
        'rag_search', {'question': 'Что такое хайку?'}, 'Что такое хайку?'
    )
    assert result is True


def test_validate_tool_call_rag_search_empty_question():
    """Validate rag_search with empty question should fail."""
    result = agent.validate_tool_call('rag_search', {'question': ''}, 'test')
    assert result is False


def test_validate_tool_call_generate_haiku():
    """Validate generate_haiku tool call."""
    result = agent.validate_tool_call(
        'generate_haiku', {'theme': 'осень'}, 'напиши хайку про осень'
    )
    assert result is True


def test_validate_tool_call_generate_haiku_empty_theme():
    """Validate generate_haiku with empty theme should fail."""
    result = agent.validate_tool_call('generate_haiku', {'theme': ''}, 'test')
    assert result is False


def test_validate_tool_call_generate_haiku_long_theme():
    """Validate generate_haiku with long theme should fail."""
    result = agent.validate_tool_call(
        'generate_haiku',
        {'theme': 'очень длинная тема для хайку которая превышает лимит'},
        'test',
    )
    assert result is False


def test_validate_tool_call_unknown_tool():
    """Validate unknown tool should fail."""
    result = agent.validate_tool_call('unknown_tool', {}, 'test')
    assert result is False


def test_detect_theme_change_with_new_theme():
    """Detect theme change with new theme specified."""
    result = agent.detect_theme_change('сменить тему на зима')
    assert result == 'зима'


def test_detect_theme_change_without_new_theme():
    """Detect theme change without new theme specified."""
    result = agent.detect_theme_change('сменить тему')
    assert result == ''


def test_detect_theme_change_no_change():
    """Detect no theme change in normal request."""
    result = agent.detect_theme_change('напиши хайку про осень')
    assert result is None


def test_print_help():
    """Print help should not raise exception."""
    agent.print_help()


def test_print_reminder():
    """Print reminder should not raise exception."""
    agent.print_reminder()
