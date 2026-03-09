## API GigaChat на ИФТ (сигма/дельта)

Для выполнения задания с API GigaChat чаще всего нужны два endpoint:

- `/chat/completions` - генерация ответа модели
- `/embeddings` - получение векторных представлений текста

### Сертификаты

Для запросов на ИФТ требуются клиентские сертификаты:
- если у вашей команды уже есть рабочий агент с сертификатами на ИФТ, просьба использовать их
- если агента нет, вы можете обратиться к автору задания за временными сертификатами

В примерах ниже предполагается, что сертификаты лежат в директории `certs/`:

- `certs/gigachat.pem` - клиентский сертификат
- `certs/gigachat.key` - приватный ключ
- `certs/chain.pem` - цепочка доверенных сертификатов (CA bundle)

### `/chat/completions`

Пример запроса:

```python
import requests

url = "https://gigachat-ift.delta.sbrf.ru/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
}
payload = {
    "model": "GigaChat",
    "messages": [
        {"role": "system", "content": "Ты полезный ассистент."},
        {"role": "user", "content": "Кратко объясни, что такое RAG."},
    ],
    "temperature": 0.7,
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    cert=("certs/gigachat.pem", "certs/gigachat.key"),
    verify="certs/chain.pem",
    timeout=30,
)
response.raise_for_status()
print(response.json())
```

## `/embeddings`

Пример запроса:

```python
import requests

url = "https://gigachat-ift.delta.sbrf.ru/v1/embeddings"
headers = {
    "Content-Type": "application/json",
}
payload = {
    "model": "Embeddings",
    "input": ["Пример текста для векторизации"],
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    cert=("certs/gigachat.pem", "certs/gigachat.key"),
    verify="certs/chain.pem",
    timeout=30,
)
response.raise_for_status()
print(response.json())
```

### Модели

Доступные модели (параметр `model` в `payload`):

- для `/chat/completions`: `GigaChat-2`, `GigaChat-2-Pro`, `GigaChat-2-Max`
- для `/embeddings`: `Embeddings`, `EmbeddingsGigaR` и др.
