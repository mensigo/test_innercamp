"""RAG tool client for poetry questions."""

import requests

from src import config
from src.utils_openai import post_chat_completions


def fetch_context(question: str, top_k: int = 3) -> list[dict]:
    """
    Fetch top chunks for the given question from RAG service.
    """
    payload = {'question': question, 'top_k': top_k}

    url = f'http://localhost:{config.tool_rag_port}/search'
    try:
        response = requests.post(url, json=payload, timeout=8)
        response.raise_for_status()
        data = response.json()
        return data.get('results', [])
    except requests.exceptions.ConnectionError:
        print('TOOL: Не удалось подключиться к RAG сервису (Connection refused)')
        return []
    except requests.exceptions.Timeout:
        print('TOOL: Превышено время ожидания ответа от RAG сервиса (Timeout)')
        return []
    except requests.exceptions.RequestException as e:
        print(f'TOOL: Ошибка при обращении к RAG сервису - {e}')
        return []
    except Exception as e:
        print(f'TOOL: Неожиданная ошибка - {e}')
        return []


def answer_question(question: str) -> str:
    """
    Answer poetry question using RAG context and LLM.
    """
    contexts = fetch_context(question)
    if not contexts:
        return 'Не удалось получить контекст из базы знаний.'

    context_lines = []
    for item in contexts:
        source = item.get('source', 'unknown')
        text = item.get('text', '')
        context_lines.append(f'Источник: {source}\n{text}')

    context_block = '\n\n'.join(context_lines)
    system_prompt = """Ты помощник по японской поэзии. Отвечай строго по контексту.
Если контекста недостаточно, скажи что данных нет."""

    user_prompt = (
        f'Вопрос: {question}\n\nКонтекст:\n{context_block}\n\nОтветь кратко и по делу.'
    )

    payload = {
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'temperature': 0.2,
    }

    response = post_chat_completions(payload)
    if 'error' in response:
        print(f'LLM Error: {response["error"]}')
        return 'Ошибка при обращении к LLM.'

    try:
        return response['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError) as e:
        print(f'LLM Response Error: {e}')
        return 'Ошибка при обработке ответа LLM.'
