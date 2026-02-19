"""Tests for tool call validation logic."""

import pytest

from src.s02_simple_haiku.validate_tool_call import validate_tool_call

pytestmark = pytest.mark.unit


def test_validate_tool_call_rag_search():
    """Validate rag_search tool call."""
    is_valid, result = validate_tool_call(
        'rag_search', {'question': 'Что такое хайку?'}
    )
    assert is_valid is True
    assert isinstance(result, dict)
    assert 'message' in result
    assert result.get('param') is None


def test_validate_tool_call_rag_search_empty_question():
    """Validate rag_search with empty question should fail."""
    is_valid, result = validate_tool_call('rag_search', {'question': ''})
    assert is_valid is False
    assert result['param'] == 'question'
    assert result['reason'] == 'empty'
    assert 'message' in result


def test_validate_tool_call_generate_haiku():
    """Validate generate_haiku tool call."""
    is_valid, result = validate_tool_call('generate_haiku', {'theme': 'осень'})
    assert is_valid is True
    assert isinstance(result, dict)
    assert 'message' in result
    assert result.get('param') is None


def test_validate_tool_call_generate_haiku_empty_theme():
    """Validate generate_haiku with empty theme should fail."""
    is_valid, result = validate_tool_call('generate_haiku', {'theme': ''})
    assert is_valid is False
    assert result['param'] == 'theme'
    assert result['reason'] == 'empty'
    assert 'message' in result


def test_validate_tool_call_generate_haiku_long_theme():
    """Validate generate_haiku with long theme should fail."""
    is_valid, result = validate_tool_call(
        'generate_haiku',
        {'theme': 'очень длинная тема для хайку которая превышает лимит'},
    )
    assert is_valid is False
    assert result['param'] == 'theme'
    assert result['reason'] == 'long'
    assert 'message' in result


def test_validate_tool_call_unknown_tool():
    """Validate unknown tool should fail."""
    is_valid, result = validate_tool_call('unknown_tool', {})
    assert is_valid is False
    assert result['param'] == 'unknown_tool'
    assert result['reason'] == 'unknown_tool'
    assert 'message' in result
