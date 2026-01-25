"""Interactive haiku-generating agent with intent classification and syllable counting."""

import sys
import requests
from pathlib import Path

from src.utils_openai import post_chat_completions


def classify_intent(user_input: str, **kwargs) -> bool:
    """
    Classify if user wants haiku/hokku generation.
    Returns True if request is for haiku, False otherwise.
    """
    system_prompt = """Ты классификатор запросов. Определи, хочет ли пользователь сгенерировать хайку или хокку.

Примеры запросов ДЛЯ ХАЙКУ (отвечай "да"):
- "напиши хайку о море"
- "сгенерируй хокку"
- "хайку про кота"
- "хоку мне о весне"
- "создай хайку"

Примеры запросов НЕ ДЛЯ ХАЙКУ (отвечай "нет"):
- "напиши стишок"
- "расскажи анекдот"
- "что такое хайку?"
- "объясни про хокку"

Отвечай ТОЛЬКО одним словом: "да" или "нет"."""

    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_input},
        ],
        'temperature': kwargs.get('temperature', 0.001),
        'max_tokens': 1,
    }

    response = post_chat_completions(payload)

    if 'error' in response:
        print(f'LLM Error: {response["error"]}')
        return False

    try:
        content = response['choices'][0]['message']['content'].strip().lower()
        return 'да' in content or 'yes' in content
    except (KeyError, IndexError) as e:
        print(f'LLM Response Error: {e}')
        return False


def extract_topic(user_input: str) -> str:
    """
    Extract topic from user request.
    Returns extracted topic or default 'окончание зимы'.
    """
    system_prompt = """Ты помощник, который извлекает тему для хайку из запроса пользователя.

Если пользователь указал конкретную тему (море, весна, кот, работа и т.д.), верни только эту тему одним-двумя словами.
Если тема не указана явно, верни: "окончание зимы"

Примеры:
- "напиши хайку о море" -> "море"
- "сгенерируй хокку про весну" -> "весна"
- "хайку про кота" -> "кот"
- "создай хайку" -> "окончание зимы"
- "хокку мне" -> "окончание зимы"

Отвечай ТОЛЬКО темой, без дополнительных слов."""

    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_input},
        ],
        'temperature': 0.001,
    }

    response = post_chat_completions(payload)

    if 'error' in response:
        print(f'LLM Error: {response["error"]}')
        return 'окончание зимы'

    try:
        topic = response['choices'][0]['message']['content'].strip()
        return topic if topic else 'окончание зимы'
    except (KeyError, IndexError) as e:
        print(f'LLM Response Error: {e}')
        return 'окончание зимы'


def generate_haiku(topic: str) -> str:
    """
    Generate Russian haiku on specified topic.
    Returns haiku text with 5-7-5 syllable structure.
    """
    system_prompt = """Ты поэт, который пишет хайку на русском языке.

СТРОГИЕ ТРЕБОВАНИЯ:
1. Формат 5-7-5 слогов (первая строка - 5 слогов, вторая - 7 слогов, третья - 5 слогов)
2. Структура:
   - Строка 1: Короткий образ (5 слогов)
   - Строка 2: Развитие образа, сопоставление (7 слогов)
   - Строка 3: Второй образ, неожиданная связь или вывод (5 слогов)
3. Используй тему, указанную пользователем
4. Пиши в стиле традиционного японского хайку
5. Выводи ТОЛЬКО текст хайку, по одной строке, БЕЗ нумерации, БЕЗ дополнительных комментариев

Пример хайку:
Море в тишине
Волны шепчут о вечном
Закат золотой"""

    user_prompt = f'Напиши хайку на тему: {topic}'

    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': 0.5,
    }

    response = post_chat_completions(payload)

    if 'error' in response:
        print(f'LLM Error: {response["error"]}')
        return ''

    try:
        haiku = response['choices'][0]['message']['content'].strip()
        return haiku
    except (KeyError, IndexError) as e:
        print(f'LLM Response Error: {e}')
        return ''


def count_syllables_via_tool(haiku_text: str) -> dict | None:
    """
    Call Flask service to count syllables in haiku.
    Returns dictionary with syllable stats or None on error.
    """
    url = 'http://localhost:8090/count'
    payload = {'text': haiku_text}

    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(
            'TOOL: Не удалось подключиться к сервису подсчета слогов (Connection refused)'
        )
        return None
    except requests.exceptions.Timeout:
        print('TOOL: Превышено время ожидания ответа от сервиса (Timeout)')
        return None
    except requests.exceptions.RequestException as e:
        print(f'TOOL: Ошибка при обращении к сервису - {e}')
        return None
    except Exception as e:
        print(f'TOOL: Неожиданная ошибка - {e}')
        return None


def display_haiku(haiku: str, stats: dict | None):
    """
    Display haiku with syllable counts if available.
    """
    if not haiku:
        print('Не удалось сгенерировать хайку.')
        return

    lines = haiku.strip().split('\n')

    print('\n--- Хайку ---')

    if stats and 'syllables_per_line' in stats:
        syllable_counts = stats['syllables_per_line']
        for i, line in enumerate(lines):
            if i < len(syllable_counts):
                print(f'{line.strip()} ({syllable_counts[i]})')
            else:
                print(line.strip())

        print(f'\nВсего слогов: {stats.get("total_syllables", "?")}')
        print(f'Всего слов: {stats.get("total_words", "?")}')
    else:
        # Display without stats if tool failed
        for line in lines:
            print(line.strip())

    print('-------------\n')


def main():
    """
    Main interactive loop for haiku agent.
    """
    print('=== Агент генерации хайку ===')
    print('Для выхода введите: exit, quit или q\n')

    while True:
        user_input = input('Введите запрос: ').strip()

        if not user_input:
            continue

        if user_input.lower() in ['exit', 'quit', 'q']:
            print('До свидания!')
            break

        # Step 1: Classify intent
        if not classify_intent(user_input):
            print('CLF: не наш агент\n')
            continue

        # Step 2: Extract topic
        topic = extract_topic(user_input)
        print(f'[Тема: {topic}]')

        # Step 3: Generate haiku
        haiku = generate_haiku(topic)

        if not haiku:
            print('Ошибка при генерации хайку.\n')
            continue

        # Step 4: Count syllables via tool
        stats = count_syllables_via_tool(haiku)

        # Step 5: Display haiku
        display_haiku(haiku, stats)


if __name__ == '__main__':
    main()
