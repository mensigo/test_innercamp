"""Extract lecturer full name from retrieved RAG chunks."""

from __future__ import annotations

from .config import config
from .logger import logger
from .utils import post_chat_completions


def _read_response_text(response: dict) -> str:
    """Extract assistant text from chat completions payload."""
    choices = response.get('choices')
    if not isinstance(choices, list) or not choices:
        return ''
    message = choices[0].get('message')
    if not isinstance(message, dict):
        return ''
    content = message.get('content')
    if not isinstance(content, str):
        return ''
    return content.strip()


def extract_lecturer(user_query: str, subject_name: str, chunks: list[str]) -> str:
    """Extract only lecturer full name from retrieved chunks."""
    if not chunks:
        return ''

    chunks_block = '\n\n'.join(
        [f'Фрагмент {idx}:\n{chunk}' for idx, chunk in enumerate(chunks, start=1)]
    )
    payload = {
        'messages': [
            {
                'role': 'system',
                'content': (
                    'Извлеки только ФИО лектора из фрагментов.\n'
                    'Правила:\n'
                    '1) Используй только текст фрагментов.\n'
                    '2) Верни строго одно ФИО без пояснений.\n'
                    '3) Не добавляй слова "лектор", скобки и знаки препинания.\n'
                    '4) Учитывай предмет запроса и выбирай ФИО лектора именно этого курса.\n'
                    '5) Если точного ФИО нет, верни пустую строку.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f'Запрос:\n{user_query}\n\n'
                    f'Канонический предмет:\n{subject_name}\n\n'
                    f'Фрагменты:\n{chunks_block}'
                ),
            },
        ],
        'temperature': config.freezing,
        'max_tokens': 80,
    }
    response = post_chat_completions(payload, verbose=config.debug)
    if 'error' in response:
        logger.warning(f'agent // lecturer extract error: {response["error"]}')
        return ''
    return _read_response_text(response)
