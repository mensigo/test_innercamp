"""GigaChat API wrapper functions."""

import requests

from src import config

DEFAULT_CHAT_MODEL = 'GigaChat-2-Max'
DEFAULT_EMBEDDINGS_MODEL = 'Embeddings'


def post_chat_completions(payload: dict, verbose: bool = False) -> dict:
    """
    Generate model response based on messages.
    Sends POST request to /chat/completions endpoint.
    """
    url = f'{config.gigachat_base_url}/chat/completions'
    if 'model' not in payload:
        payload['model'] = DEFAULT_CHAT_MODEL
    try:
        if verbose:
            print('req:', payload)
        response = requests.post(
            url,
            json=payload,
            cert=(config.gigachat_cert_path, config.gigachat_key_path),
            verify=config.gigachat_chain_path,
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
    url = f'{config.gigachat_base_url}/embeddings'
    if 'model' not in payload:
        payload['model'] = DEFAULT_EMBEDDINGS_MODEL
    try:
        response = requests.post(
            url,
            json=payload,
            cert=(config.gigachat_cert_path, config.gigachat_key_path),
            verify=config.gigachat_chain_path,
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
    url = f'{config.gigachat_base_url}/models'
    try:
        response = requests.get(
            url,
            cert=(config.gigachat_cert_path, config.gigachat_key_path),
            verify=config.gigachat_chain_path,
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
    url = f'{config.gigachat_base_url}/files'
    try:
        response = requests.get(
            url,
            cert=(config.gigachat_cert_path, config.gigachat_key_path),
            verify=config.gigachat_chain_path,
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
    url = f'{config.gigachat_base_url}/files'
    try:
        response = requests.post(
            url,
            json=payload,
            cert=(config.gigachat_cert_path, config.gigachat_key_path),
            verify=config.gigachat_chain_path,
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
    url = f'{config.gigachat_base_url}/files/{file_id}/delete'
    try:
        response = requests.post(
            url,
            cert=(config.gigachat_cert_path, config.gigachat_key_path),
            verify=config.gigachat_chain_path,
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
    url = f'{config.gigachat_base_url}/files/{file_id}/content'
    try:
        response = requests.get(
            url,
            cert=(config.gigachat_cert_path, config.gigachat_key_path),
            verify=config.gigachat_chain_path,
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
    url = f'{config.gigachat_base_url}/files/{file_id}'
    try:
        response = requests.get(
            url,
            cert=(config.gigachat_cert_path, config.gigachat_key_path),
            verify=config.gigachat_chain_path,
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
    url = f'{config.gigachat_base_url}/tokens/count'
    try:
        response = requests.post(
            url,
            json=payload,
            cert=(config.gigachat_cert_path, config.gigachat_key_path),
            verify=config.gigachat_chain_path,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}
