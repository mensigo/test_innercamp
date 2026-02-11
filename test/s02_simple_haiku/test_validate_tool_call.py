"""Tests for tool call validation logic."""

import pytest

from src.s02_simple_haiku.validate_tool_call import (
    detect_theme_change,
    validate_tool_call,
)

pytestmark = pytest.mark.unit


def test_validate_tool_call_rag_search():
    """Validate rag_search tool call."""
    is_valid, missing_param = validate_tool_call(
        'rag_search', {'question': 'Что такое хайку?'}, 'Что такое хайку?'
    )
    assert is_valid is True
    assert missing_param is None


def test_validate_tool_call_rag_search_empty_question():
    """Validate rag_search with empty question should fail."""
    is_valid, missing_param = validate_tool_call('rag_search', {'question': ''}, 'test')
    assert is_valid is False
    assert missing_param == 'question'


def test_validate_tool_call_generate_haiku():
    """Validate generate_haiku tool call."""
    is_valid, missing_param = validate_tool_call(
        'generate_haiku', {'theme': 'осень'}, 'напиши хайку про осень'
    )
    assert is_valid is True
    assert missing_param is None


def test_validate_tool_call_generate_haiku_empty_theme():
    """Validate generate_haiku with empty theme should fail."""
    is_valid, missing_param = validate_tool_call(
        'generate_haiku', {'theme': ''}, 'test'
    )
    assert is_valid is False
    assert missing_param == 'theme'


def test_validate_tool_call_generate_haiku_long_theme():
    """Validate generate_haiku with long theme should fail."""
    is_valid, missing_param = validate_tool_call(
        'generate_haiku',
        {'theme': 'очень длинная тема для хайку которая превышает лимит'},
        'test',
    )
    assert is_valid is False
    assert missing_param is None  # Ошибка валидации, но не missing param


def test_validate_tool_call_unknown_tool():
    """Validate unknown tool should fail."""
    is_valid, missing_param = validate_tool_call('unknown_tool', {}, 'test')
    assert is_valid is False
    assert missing_param is None


def test_detect_theme_change_with_new_theme():
    """Detect theme change with new theme specified."""
    result = detect_theme_change('сменить тему на зима')
    assert result == 'зима'


def test_detect_theme_change_without_new_theme():
    """Detect theme change without new theme specified."""
    result = detect_theme_change('сменить тему')
    assert result == ''


def test_detect_theme_change_no_change():
    """Detect no theme change in normal request."""
    result = detect_theme_change('напиши хайку про осень')
    assert result is None
