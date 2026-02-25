# Запросы на LLM API

## 1. Запрос на генерацию /chat/completions

Отдельная функция: `post_chat_completions(payload: dict, verbose: bool = False) -> dict`

- `payload` обязательно содержит `messages`, может содержать доп. параметры (напр., `temperature`)
- если `payload` не содержит `model`, то устанавливается дефолтное значение `config.default_model` 
- если `verbose=True`, то в лог пишется `payload` (до запроса) и `response` (после успешного запроса) на уровне debug
- если при запросе происходит ошибка, то в лог пишется стектрейс ошибки на уровне error/critical и возвращается словарь вида `{"error": "<exception text>"}`
- если запрос проходит успешно, возвращается стандартный ответ вида [result_chatcomp.json](../meta/gigachat/result_chatcomp.json)

## 2. Запрос на эмбеддинги /embeddings

Отдельная функция: `post_embeddings(payload: dict, verbose: bool = False) -> dict`

- `payload` обязательно содержит `input`
- если `payload` не содержит `model`, то устанавливается дефолтное значение `config.default_embedding_model` 
- если `verbose=True`, то в лог пишется `payload` (до запроса) и `response` (после успешного запроса) на уровне debug
- если при запросе происходит ошибка, то в лог пишется стектрейс ошибки на уровне error/critical и возвращается словарь вида `{"error": "< exception text >"}`
- если запрос проходит успешно, возвращается стандартный ответ вида [result_embeddings.json](../meta/gigachat/result_embeddings.json)
