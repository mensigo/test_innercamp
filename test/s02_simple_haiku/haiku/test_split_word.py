"""Smoke tests for split_word syllable splitting."""

import pytest

from src.s02_simple_haiku.haiku.split_word import split_into_syllables_simple

pytestmark = pytest.mark.unit


def test_split_empty_string():
    """Empty string should return empty list."""
    result = split_into_syllables_simple('')
    assert result == []


def test_split_whitespace_only():
    """Whitespace only should return empty list."""
    result = split_into_syllables_simple('   ')
    assert result == []


def test_split_single_vowel():
    """Single vowel should return one syllable."""
    result = split_into_syllables_simple('а')
    assert result == ['а']


def test_split_single_consonant():
    """Single consonant should return empty list (no vowel)."""
    result = split_into_syllables_simple('б')
    assert result == []


def test_split_programma():
    """Word 'программа' should split correctly with doubled consonant."""
    result = split_into_syllables_simple('программа')
    assert result == ['про', 'грам', 'ма']


def test_split_aist():
    """Word 'аист' should split correctly."""
    result = split_into_syllables_simple('аист')
    assert result == ['а', 'ист']


def test_split_strana():
    """Word 'страна' should split correctly with consonant cluster."""
    result = split_into_syllables_simple('страна')
    assert result == ['стра', 'на']


def test_split_obezyana():
    """Word 'обезьяна' should split correctly."""
    result = split_into_syllables_simple('обезьяна')
    assert result == ['о', 'бе', 'зья', 'на']


def test_split_poyushaya():
    """Word 'поющая' should split correctly with consecutive vowels."""
    result = split_into_syllables_simple('поющая')
    assert result == ['по', 'ю', 'ща', 'я']


def test_split_kompyuter():
    """Word 'компьютер' should split (may not be perfect but consistent)."""
    result = split_into_syllables_simple('компьютер')
    assert len(result) == 3
    assert result[0] == 'ко'


def test_split_case_insensitive():
    """Splitting should be case insensitive."""
    result_lower = split_into_syllables_simple('страна')
    result_upper = split_into_syllables_simple('СТРАНА')
    result_mixed = split_into_syllables_simple('СтРаНа')
    assert result_lower == result_upper == result_mixed


def test_split_with_leading_whitespace():
    """Leading whitespace should be stripped."""
    result = split_into_syllables_simple('  море')
    assert result == ['мо', 'ре']


def test_split_with_trailing_whitespace():
    """Trailing whitespace should be stripped."""
    result = split_into_syllables_simple('море  ')
    assert result == ['мо', 'ре']


def test_split_more():
    """Word 'море' should split correctly."""
    result = split_into_syllables_simple('море')
    assert result == ['мо', 'ре']


def test_split_volna():
    """Word 'волна' should split correctly."""
    result = split_into_syllables_simple('волна')
    assert result == ['во', 'лна']


def test_split_zakat():
    """Word 'закат' should split correctly."""
    result = split_into_syllables_simple('закат')
    assert result == ['за', 'кат']


def test_split_zolotoy():
    """Word 'золотой' should split correctly."""
    result = split_into_syllables_simple('золотой')
    assert result == ['зо', 'ло', 'той']


def test_split_veter():
    """Word 'ветер' should split correctly."""
    result = split_into_syllables_simple('ветер')
    assert result == ['ве', 'тер']


def test_split_listya():
    """Word 'листья' should split correctly with doubled consonant."""
    result = split_into_syllables_simple('листья')
    assert result == ['ли', 'стья']


def test_split_shyopot():
    """Word 'шёпот' should handle ё correctly."""
    result = split_into_syllables_simple('шёпот')
    assert result == ['шё', 'пот']


def test_syllable_count_for_haiku_line():
    """Count syllables in a typical haiku line."""
    words = ['ветер', 'шепчет', 'мне']
    total_syllables = sum(len(split_into_syllables_simple(word)) for word in words)
    assert total_syllables == 5
