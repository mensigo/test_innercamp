## s02_simple_haiku

Мини-агент с CLI для японской поэзии:

- Ответы на вопросы о хайку/хокку через RAG (Flask + FAISS)
- Генерация хайку по теме

### Запуск сервисов

1) Сервис подсчета слогов:

```
python -m src.s02_simple_haiku.haiku.tool_count_service
```

2) RAG сервис:

```
python -m src.s02_simple_haiku.rag.rag_service
```

### Запуск агента

```
python -m src.s02_simple_haiku.agent
```

### Данные для RAG

Markdown-файлы лежат в `src/s02_simple_haiku/rag/data/`.
