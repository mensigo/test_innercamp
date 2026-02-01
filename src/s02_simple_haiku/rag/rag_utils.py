"""RAG tool client for poetry questions."""

import requests

from src import config


def check_health() -> bool:
    """
    Check RAG service health endpoint.
    """
    url = f'http://localhost:{config.tool_rag_port}/health'
    try:
        response = requests.get(url, timeout=4)
        response.raise_for_status()
        data = response.json()
        return bool(data.get('index_ready'))
    except requests.exceptions.ConnectionError:
        print('RAG-TOOL: Не удалось подключиться к RAG сервису (Connection refused)')
        return False
    except requests.exceptions.Timeout:
        print('RAG-TOOL: Превышено время ожидания ответа от RAG сервиса (Timeout)')
        return False
    except requests.exceptions.RequestException as e:
        print(f'RAG-TOOL: Ошибка при обращении к RAG сервису - {e}')
        return False
    except Exception as e:
        print(f'RAG-TOOL: Неожиданная ошибка - {e}')
        return False


def fetch_answer(question: str, top_k: int = 2) -> dict:
    """
    Fetch answer and chunk metadata from RAG service.
    """
    if not check_health():
        return {}

    payload = {'question': question, 'top_k': top_k}

    url = f'http://localhost:{config.tool_rag_port}/search'
    try:
        response = requests.post(url, json=payload, timeout=8)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print('RAG-TOOL: Не удалось подключиться к RAG сервису (Connection refused)')
        return {}
    except requests.exceptions.Timeout:
        print('RAG-TOOL: Превышено время ожидания ответа от RAG сервиса (Timeout)')
        return {}
    except requests.exceptions.RequestException as e:
        print(f'RAG-TOOL: Ошибка при обращении к RAG сервису - {e}')
        return {}
    except Exception as e:
        print(f'RAG-TOOL: Неожиданная ошибка - {e}')
        return {}


def answer_question(question: str) -> dict:
    """
    Answer poetry question using RAG service.
    """
    response = fetch_answer(question)
    if not response:
        return {
            'answer': 'Не удалось получить ответ от RAG сервиса.',
            'chunk_title_list': [],
            'chunk_texts': [],
        }
    if 'error' in response:
        return {
            'answer': str(response.get('error')),
            'chunk_title_list': [],
            'chunk_texts': [],
        }
    return {
        'answer': response.get('answer', ''),
        'chunk_title_list': response.get('chunk_title_list', []),
        'chunk_texts': response.get('chunk_texts', []),
    }
