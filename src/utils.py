"""GigaChat API wrapper functions."""

import os

import requests
from dotenv import load_dotenv


load_dotenv()

# Global configuration
GIGACHAT_BASE_URL = os.getenv(
    'GIGACHAT_BASE_URL', 'https://gigachat.devices.sberbank.ru/api/v1'
)
GIGACHAT_CERT_PATH = os.getenv('GIGACHAT_CERT_PATH', '/path/to/cert.pem')
GIGACHAT_KEY_PATH = os.getenv('GIGACHAT_KEY_PATH', '/path/to/key.pem')
GIGACHAT_CHAIN_PATH = os.getenv('GIGACHAT_CHAIN_PATH', '/path/to/ca_bundle.pem')
DEFAULT_CHAT_MODEL = 'GigaChat-2-Max'
DEFAULT_EMBEDDINGS_MODEL = 'Embeddings'

assert os.path.exists(GIGACHAT_CERT_PATH), f'wrong {GIGACHAT_CERT_PATH=}'
assert os.path.exists(GIGACHAT_KEY_PATH), f'wrong {GIGACHAT_KEY_PATH=}'
assert os.path.exists(GIGACHAT_CHAIN_PATH), f'wrong {GIGACHAT_CHAIN_PATH=}'


def post_chat_completions(payload: dict, verbose: bool = False) -> dict:
    """
    Generate model response based on messages.
    Sends POST request to /chat/completions endpoint.
    """
    url = f'{GIGACHAT_BASE_URL}/chat/completions'
    if 'model' not in payload:
        payload['model'] = DEFAULT_CHAT_MODEL
    try:
        if verbose:
            print('req:', payload)
        response = requests.post(
            url,
            json=payload,
            cert=(GIGACHAT_CERT_PATH, GIGACHAT_KEY_PATH),
            verify=GIGACHAT_CHAIN_PATH,
        )
        if verbose:
            print('ans:', response, response.text)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def post_embeddings(payload: dict) -> dict:
    """
    Create vector embeddings for text.
    Sends POST request to /embeddings endpoint.
    """
    url = f'{GIGACHAT_BASE_URL}/embeddings'
    if 'model' not in payload:
        payload['model'] = DEFAULT_EMBEDDINGS_MODEL
    try:
        response = requests.post(
            url,
            json=payload,
            cert=(GIGACHAT_CERT_PATH, GIGACHAT_KEY_PATH),
            verify=GIGACHAT_CHAIN_PATH,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def get_models() -> dict:
    """
    Retrieve list of available models.
    Sends GET request to /models endpoint.
    """
    url = f'{GIGACHAT_BASE_URL}/models'
    try:
        response = requests.get(
            url,
            cert=(GIGACHAT_CERT_PATH, GIGACHAT_KEY_PATH),
            verify=GIGACHAT_CHAIN_PATH,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def get_files_list() -> dict:
    """
    Get list of uploaded files.
    Sends GET request to /files endpoint.
    """
    url = f'{GIGACHAT_BASE_URL}/files'
    try:
        response = requests.get(
            url,
            cert=(GIGACHAT_CERT_PATH, GIGACHAT_KEY_PATH),
            verify=GIGACHAT_CHAIN_PATH,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def post_files(payload: dict) -> dict:
    """
    Upload file to storage.
    Sends POST request to /files endpoint.
    """
    url = f'{GIGACHAT_BASE_URL}/files'
    try:
        response = requests.post(
            url,
            json=payload,
            cert=(GIGACHAT_CERT_PATH, GIGACHAT_KEY_PATH),
            verify=GIGACHAT_CHAIN_PATH,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def post_files_delete(file_id: str) -> dict:
    """
    Delete file from storage.
    Sends POST request to /files/{file_id}/delete endpoint.
    """
    url = f'{GIGACHAT_BASE_URL}/files/{file_id}/delete'
    try:
        response = requests.post(
            url,
            cert=(GIGACHAT_CERT_PATH, GIGACHAT_KEY_PATH),
            verify=GIGACHAT_CHAIN_PATH,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def get_files_content(file_id: str) -> dict:
    """
    Retrieve file content.
    Sends GET request to /files/{file_id}/content endpoint.
    """
    url = f'{GIGACHAT_BASE_URL}/files/{file_id}/content'
    try:
        response = requests.get(
            url,
            cert=(GIGACHAT_CERT_PATH, GIGACHAT_KEY_PATH),
            verify=GIGACHAT_CHAIN_PATH,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def get_files_info(file_id: str) -> dict:
    """
    Get metadata for specific file.
    Sends GET request to /files/{file_id} endpoint.
    """
    url = f'{GIGACHAT_BASE_URL}/files/{file_id}'
    try:
        response = requests.get(
            url,
            cert=(GIGACHAT_CERT_PATH, GIGACHAT_KEY_PATH),
            verify=GIGACHAT_CHAIN_PATH,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def get_tokens_count(payload: dict) -> dict:
    """
    Count tokens in provided texts.
    Sends POST request to /tokens/count endpoint.
    """
    url = f'{GIGACHAT_BASE_URL}/tokens/count'
    try:
        response = requests.post(
            url,
            json=payload,
            cert=(GIGACHAT_CERT_PATH, GIGACHAT_KEY_PATH),
            verify=GIGACHAT_CHAIN_PATH,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}
