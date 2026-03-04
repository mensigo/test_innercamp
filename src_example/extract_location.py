"""Extract lecture location from retrieved RAG chunks."""

from __future__ import annotations

from .config import config
from .logger import logger
from .utils import post_chat_completions

LOCATION_SYSTEM_PROMPT_TEMPLATE = """
Ты извлекаешь место проведения ЛЕКЦИЙ из фрагментов.
Используй только данные из фрагментов.
Нужный курс: {subject_name}.
Алгоритм:
1) Сначала отфильтруй фрагменты по точному названию курса (первая непустая строка):
   - "Machine Learning" -> "Машинное обучение"
   - "Probability Theory" -> "Теория вероятности"
   - "Optimization Theory" -> "Методы оптимизации"
2) Запрещено брать данные из похожих курсов:
   - "Введение в машинное обучение"
   - "Машинное обучение на практике"
   - "Машинное обучение на больших объемах данных"
   - "Теория вероятностей и математическая статистика"
   - "Непрерывная оптимизация"
   - "Методы оптимизации в машинном обучении"
3) Игнорируй любые аудитории из семинаров, практик, экзаменов и блоков
   "Распределение по аудиториям". Нужна только аудитория лекции.
4) Для "Probability Theory" аудиторию бери только из строки таблицы,
   где в колонке "Тип занятия" стоит "лекция".

Верни строго один токен аудитории без пояснений, см. примеры:
- корректно: "П3", "П5", "П6а", "R200", "R301", "R402"
- некорректно: "ауд. П3", "лекция в аудитории R200", "по средам в R402"
Если точного места лекций нет, верни пустую строку.
""".strip()


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


def _canonical_title(subject_name: str) -> str:
    """Return strict chunk title for canonical subject."""
    return {
        'Machine Learning': 'Машинное обучение',
        'Probability Theory': 'Теория вероятности',
        'Optimization Theory': 'Методы оптимизации',
    }.get(subject_name, '')


def _first_nonempty_line(chunk: str) -> str:
    """Return first non-empty line from chunk."""
    for line in chunk.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ''


def extract_location(user_query: str, subject_name: str, chunks: list[str]) -> str:
    """Extract lecture location for the requested subject from chunks."""
    if not chunks:
        return ''

    title = _canonical_title(subject_name)
    subject_chunks = [chunk for chunk in chunks if _first_nonempty_line(chunk) == title]
    chunks_to_use = subject_chunks or chunks

    chunks_block = '\n\n'.join(
        [
            f'Фрагмент {idx}:\n{chunk}'
            for idx, chunk in enumerate(chunks_to_use, start=1)
        ]
    )
    payload = {
        'messages': [
            {
                'role': 'system',
                'content': LOCATION_SYSTEM_PROMPT_TEMPLATE.format(
                    subject_name=subject_name
                ),
            },
            {
                'role': 'user',
                'content': f'Вопрос:\n{user_query}\n\nФрагменты:\n{chunks_block}',
            },
        ],
        'temperature': config.freezing,
        'max_tokens': 20,
    }
    response = post_chat_completions(payload, verbose=config.debug)
    if 'error' in response:
        logger.warning(f'agent // location extract error: {response["error"]}')
        return ''

    raw_answer = _read_response_text(response)
    if not raw_answer:
        return ''
    for line in raw_answer.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ''
