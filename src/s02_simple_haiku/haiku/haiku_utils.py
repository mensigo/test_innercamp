"""Client for haiku generation service."""

import requests

from src import config


def check_health() -> bool:
    """
    Check haiku service health endpoint.
    """
    url = f'http://localhost:{config.tool_haiku_port}/health'
    try:
        response = requests.get(url, timeout=4)
        response.raise_for_status()
        data = response.json()
        return data.get('status') == 'ok'
    except requests.exceptions.ConnectionError:
        print(
            'HAIKU-TOOL: Не удалось подключиться к haiku сервису (Connection refused)'
        )
        return False
    except requests.exceptions.Timeout:
        print('HAIKU-TOOL: Превышено время ожидания ответа от haiku сервиса (Timeout)')
        return False
    except requests.exceptions.RequestException as e:
        print(f'HAIKU-TOOL: Ошибка при обращении к haiku сервису - {e}')
        return False
    except Exception as e:
        print(f'HAIKU-TOOL: Неожиданная ошибка - {e}')
        return False


def generate_haiku(topic: str) -> dict:
    """
    Generate haiku using haiku service.
    """
    error_response = {
        'haiku_text': '',
        'syllables_per_line': [],
        'total_words': 0,
        'error': '',
    }

    if not check_health():
        error_response['error'] = 'Service not available'
        return error_response

    url = f'http://localhost:{config.tool_haiku_port}/generate_haiku'
    payload = {'topic': topic}

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(
            'HAIKU-TOOL: Не удалось подключиться к haiku сервису (Connection refused)'
        )
        error_response['error'] = 'Connection refused'
        return error_response
    except requests.exceptions.Timeout:
        print('HAIKU-TOOL: Превышено время ожидания ответа от haiku сервиса (Timeout)')
        error_response['error'] = 'Timeout'
        return error_response
    except requests.exceptions.RequestException as e:
        print(f'HAIKU-TOOL: Ошибка при обращении к haiku сервису - {e}')
        error_response['error'] = str(e)
        return error_response
    except Exception as e:
        print(f'HAIKU-TOOL: Неожиданная ошибка - {e}')
        error_response['error'] = str(e)
        return error_response
