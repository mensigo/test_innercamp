## utils

Запросы на эндпоинты /chat/completions, /embeddings.

## s01_api_discover

Применение параметра profanity_check (gigachat).

## s02_simple_haiku

Мини-агент с CLI для японской поэзии:

- Ответы на вопросы о хайку/хокку через RAG (Flask + FAISS)
- Генерация хайку по теме


## current state

Работает (есть тесты):
- utils
- s01_api_discover (profanity_check)
- s02_simple_haiku (rag)

TODO:
- s02_simple_haiku (haiku generation tool, agent)
