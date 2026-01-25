"""GigaChat API wrapper functions."""

import requests
from typing import Optional


# Global configuration
GIGACHAT_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
GIGACHAT_CERT_PATH = "/path/to/cert.pem"  # Placeholder for SSL certificate
GIGACHAT_KEY_PATH = "/path/to/key.pem"  # Placeholder for SSL key


def post_chat_completions(payload: dict) -> dict:
    """
    Generate model response based on messages.
    Sends POST request to /chat/completions endpoint.
    """
    url = f"{GIGACHAT_BASE_URL}/chat/completions"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def post_embeddings(payload: dict) -> dict:
    """
    Create vector embeddings for text.
    Sends POST request to /embeddings endpoint.
    """
    url = f"{GIGACHAT_BASE_URL}/embeddings"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_models() -> dict:
    """
    Retrieve list of available models.
    Sends GET request to /models endpoint.
    """
    url = f"{GIGACHAT_BASE_URL}/models"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_files_list() -> dict:
    """
    Get list of uploaded files.
    Sends GET request to /files endpoint.
    """
    url = f"{GIGACHAT_BASE_URL}/files"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def post_files(payload: dict) -> dict:
    """
    Upload file to storage.
    Sends POST request to /files endpoint.
    """
    url = f"{GIGACHAT_BASE_URL}/files"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def post_files_delete(file_id: str) -> dict:
    """
    Delete file from storage.
    Sends POST request to /files/{file_id}/delete endpoint.
    """
    url = f"{GIGACHAT_BASE_URL}/files/{file_id}/delete"
    try:
        response = requests.post(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_files_content(file_id: str) -> dict:
    """
    Retrieve file content.
    Sends GET request to /files/{file_id}/content endpoint.
    """
    url = f"{GIGACHAT_BASE_URL}/files/{file_id}/content"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_files_info(file_id: str) -> dict:
    """
    Get metadata for specific file.
    Sends GET request to /files/{file_id} endpoint.
    """
    url = f"{GIGACHAT_BASE_URL}/files/{file_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def get_tokens_count(payload: dict) -> dict:
    """
    Count tokens in provided texts.
    Sends POST request to /tokens/count endpoint.
    """
    url = f"{GIGACHAT_BASE_URL}/tokens/count"
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
