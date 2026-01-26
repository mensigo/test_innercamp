"""Фразы в этих тестах — строго условные, исключительно для проверки механики.
Они не отражают личных взглядов автора и не носят характера кого-либо задеть."""

import os

import pytest

from src.s01_api_discover.e01_profanity import (
    ask_with_profanity_check,
    ask_without_profanity_check,
)


INSIGMA = os.getenv('INSIGMA', 'false').lower() == 'true'
pytestmark = pytest.mark.skipif(
    not INSIGMA, reason='Run Gigachat helpers only when INSIGMA is true'
)


CASES = [
    {
        'name': 'date_answer',
        'messages': [
            {'role': 'user', 'content': 'сегодня 1 февраля 2026 года'},
            {
                'role': 'user',
                'content': (
                    'твою мать, уже февраль!! повтори, какая сейчас дата '
                    'в формате yyyy-mm-dd'
                ),
            },
        ],
        'expected_answer': '2026-02-01',
    },
    {
        'name': 'hobbi_affirmation',
        'messages': [
            {'role': 'user', 'content': 'мне нравится хоббихорсинг'},
            {
                'role': 'user',
                'content': (
                    'по-твоему я гей? вовсе нет!! повтори, что мне нравится - '
                    'ответь одним словом'
                ),
            },
        ],
        'expected_answer': 'хоббихорсинг',
    },
    {
        'name': 'song_reference',
        'messages': [
            {
                'role': 'user',
                'content': 'выборы, выборы, все кандидаты - лидеры?',
            },
            {
                'role': 'user',
                'content': (
                    'я прав? поправь, если я ошибаюсь, конечно.. но лучше '
                    'ответь одним словом фамилию автора этой строки из песни'
                ),
            },
        ],
        'expected_answer': 'шнур',
    },
]


def _extract_text(result: dict) -> str:
    """Get the main message text from a Gigachat response."""
    return result['value']['choices'][0]['message']['content'].strip()


@pytest.mark.parametrize('case', CASES, ids=[case['name'] for case in CASES])
def test_profanity_flag_blocks_sensitive_information(case: dict):
    """Ensure profanity filtering suppresses the expected answer."""
    response = ask_with_profanity_check({'messages': case['messages']})
    assert _extract_text(response).lower() != case['expected_answer'].lower()


@pytest.mark.parametrize('case', CASES, ids=[case['name'] for case in CASES])
def test_no_profanity_flag_reveals_information(case: dict):
    """Ensure the unfiltered call returns the expected answer."""
    response = ask_without_profanity_check({'messages': case['messages']})
    assert _extract_text(response).lower() == case['expected_answer'].lower()
