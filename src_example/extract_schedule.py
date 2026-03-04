"""Extract lecture schedule answer from retrieved RAG chunks."""

from __future__ import annotations

from .config import config
from .logger import logger
from .utils import post_chat_completions

SCHEDULE_SYSTEM_PROMPT_TEMPLATE = """
Ты извлекаешь расписание ЛЕКЦИЙ из фрагментов.
Используй только данные из фрагментов.
Нужный курс: {subject_name}.
Игнорируй семинары и экзамены.
Сначала выбери только фрагменты нужного курса, потом извлекай расписание.
Не смешивай данные из разных курсов.
Если есть таблица расписания, бери только строку, где Тип занятия = лекция.
Если есть несколько курсов, выбери только вариант нужного курса.
Жесткая фильтрация по названию курса:
- Для "Machine Learning" используй только курс с точным названием "Машинное обучение".
- Для "Machine Learning" первая непустая строка фрагмента должна быть ровно "Машинное обучение".
- Не используй фрагменты "Введение в машинное обучение", "Машинное обучение на практике", "Машинное обучение на больших объемах данных".
Если нужен курс "Probability Theory", используй только фрагменты про "Теория вероятности" и не используй фрагменты про "Теория вероятностей и математическая статистика".
- Для "Probability Theory" первая непустая строка фрагмента должна быть ровно "Теория вероятности".
- Для "Optimization Theory" используй только фрагменты про "Методы оптимизации" и не используй фрагменты про "Непрерывная оптимизация" и "Методы оптимизации в машинном обучении".
- Для "Optimization Theory" первая непустая строка фрагмента должна быть ровно "Методы оптимизации".
Если во фрагменте нет точного названия нужного курса, такой фрагмент нельзя использовать.
Верни только одну строку без пояснений.
Нормализация:
- Для "Machine Learning" и "Probability Theory": если есть день недели и интервал времени, верни: "по <день во множественном числе>, HH:MM - HH:MM".
- Для "Optimization Theory": если формат типа "вторник, лекция в 13:00", верни "по вторникам, в 13:00".
- Для "Optimization Theory" не возвращай формат диапазона "HH:MM - HH:MM", возвращай только начало лекции.
- Примеры: "среда, с 10:30 до 12:00" -> "по средам, 10:30 - 12:00"; "по вторникам, 09:00 - 10:30" -> "по вторникам, 09:00 - 10:30".
- Пример для оптимизации: "вторник, лекция в 13:00" -> "по вторникам, в 13:00".
- Если указан уже корректный формат, сохрани его.
- Если точного расписания лекции нет, верни пустую строку.
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


def extract_schedule(user_query: str, subject_name: str, chunks: list[str]) -> str:
    """Extract and normalize lecture schedule from retrieved chunks."""
    if not chunks:
        return ''

    chunks_block = '\n\n'.join(
        [f'Фрагмент {idx}:\n{chunk}' for idx, chunk in enumerate(chunks, start=1)]
    )
    payload = {
        'messages': [
            {
                'role': 'system',
                'content': SCHEDULE_SYSTEM_PROMPT_TEMPLATE.format(
                    subject_name=subject_name
                ),
            },
            {
                'role': 'user',
                'content': f'Вопрос:\n{user_query}\n\nФрагменты:\n{chunks_block}',
            },
        ],
        'temperature': config.freezing,
        'max_tokens': 30,
    }
    response = post_chat_completions(payload, verbose=config.debug)
    if 'error' in response:
        logger.warning(f'agent // schedule extract error: {response["error"]}')
        return ''

    raw_answer = _read_response_text(response)
    if not raw_answer:
        return ''

    for line in raw_answer.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ''
