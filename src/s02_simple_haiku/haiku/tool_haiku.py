"""Tools for haiku generation and syllable counting."""

import requests

from src.utils_openai import post_chat_completions


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
            'TOOL: Не удалось подключиться к сервису подсчета слогов '
            '(Connection refused)'
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
