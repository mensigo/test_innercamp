# ruff: noqa: RUF001
"""Smoke tests for split_word syllable splitting."""

import pytest

from src.s02_simple_haiku.haiku.count_stats import (
    count_syllables_and_words,
    split_into_syllables_simple,
)

pytestmark = [pytest.mark.unit, pytest.mark.haiku]


def test_split_empty_string():
    """Empty string should return empty list."""
    result = split_into_syllables_simple('')
    assert len(result) == 0


def test_split_whitespace_only():
    """Whitespace only should return empty list."""
    result = split_into_syllables_simple('   ')
    assert len(result) == 0


@pytest.mark.parametrize(
    ('word', 'expected_count'),
    [
        ('а', 1),
        ('б', 0),
        ('программа', 3),
        ('аист', 2),
        ('страна', 2),
        ('обезьяна', 4),
        ('поющая', 4),
        ('компьютер', 3),
        ('море', 2),
        ('волна', 2),
        ('закат', 2),
        ('золотой', 3),
        ('ветер', 2),
        ('листья', 2),
        ('шёпот', 2),
    ],
)
def test_split_word_counts(word: str, expected_count: int):
    """Different words should produce expected syllable counts."""
    result = split_into_syllables_simple(word)
    assert len(result) == expected_count


def test_split_case_insensitive():
    """Splitting should be case insensitive."""
    result_lower = split_into_syllables_simple('страна')
    result_upper = split_into_syllables_simple('СТРАНА')
    result_mixed = split_into_syllables_simple('СтРаНа')
    assert len(result_lower) == len(result_upper) == len(result_mixed) == 2


def test_split_with_leading_whitespace():
    """Leading whitespace should be stripped."""
    result = split_into_syllables_simple('  море')
    assert len(result) == 2


def test_split_with_trailing_whitespace():
    """Trailing whitespace should be stripped."""
    result = split_into_syllables_simple('море  ')
    assert len(result) == 2


def test_syllable_count_for_haiku_line():
    """Count syllables in a typical haiku line."""
    words = ['ветер', 'шепчет', 'мне']
    total_syllables = sum(len(split_into_syllables_simple(word)) for word in words)
    assert total_syllables == 5


@pytest.mark.parametrize(
    ('text', 'expected_syllables', 'expected_words'),
    [
        ('ветер шепчет мне\nморе в тишине\nзакат золотой', [5, 5, 5], 8),
        (
            'программа аист\nстрана обезьяна\nпоющая компьютер',
            [5, 6, 7],
            6,
        ),
        (
            'море, волна!\n\nзолотой ветер',
            [4, 5],
            4,
        ),
    ],
)
def test_count_syllables_and_words(
    text: str, expected_syllables: list[int], expected_words: int
):
    """Syllable and word stats should match splitter expectations."""
    stats = count_syllables_and_words(text)

    assert stats['syllables_per_line'] == expected_syllables
    assert stats['total_words'] == expected_words
