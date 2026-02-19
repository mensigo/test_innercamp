"""Tests for parameter clarification logic."""

import pytest
from src.s02_simple_haiku.clarify import (
    extract_param_from_clarification,
    generate_clarification_prompt,
)

pytestmark = pytest.mark.unit


def test_generate_clarification_prompt_theme():
    """Generate clarification prompt for missing theme."""
    prompt = generate_clarification_prompt('generate_haiku', 'theme')
    assert 'тему' in prompt.lower()
    assert 'хайку' in prompt.lower()


def test_generate_clarification_prompt_question():
    """Generate clarification prompt for missing question."""
    prompt = generate_clarification_prompt('rag_search', 'question')
    assert 'вопрос' in prompt.lower()


def test_extract_param_from_clarification_theme_simple():
    """Extract theme from simple clarification."""
    result = extract_param_from_clarification('зима', 'theme', 'generate_haiku')
    assert result == 'зима'


def test_extract_param_from_clarification_theme_with_pro():
    """Extract theme from clarification with 'про'."""
    result = extract_param_from_clarification('про море', 'theme', 'generate_haiku')
    assert result == 'море'


def test_extract_param_from_clarification_theme_too_long():
    """Extract theme that is too long should return None."""
    result = extract_param_from_clarification(
        'очень длинная тема для хайку которая превышает лимит',
        'theme',
        'generate_haiku',
    )
    assert result is None


def test_extract_param_from_clarification_theme_ignore_phrases():
    """Extract theme with ignore phrases should return None."""
    assert (
        extract_param_from_clarification('не знаю', 'theme', 'generate_haiku') is None
    )
    assert extract_param_from_clarification('хз', 'theme', 'generate_haiku') is None
    assert (
        extract_param_from_clarification('что-нибудь', 'theme', 'generate_haiku')
        is None
    )


def test_extract_param_from_clarification_question():
    """Extract question from clarification."""
    result = extract_param_from_clarification(
        'Что такое хайку?', 'question', 'rag_search'
    )
    assert result == 'Что такое хайку?'


def test_extract_param_from_clarification_question_too_short():
    """Extract question that is too short should return None."""
    result = extract_param_from_clarification('да', 'question', 'rag_search')
    assert result is None
