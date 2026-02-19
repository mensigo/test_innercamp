## Пример реализации

### 1. build_index

build_rag_chunks - чтение Markdown файлов из директории, чанкирование  
init_faiss_index - построение индекса по набору чанков, сохранение индекса (опц. загрузка сохраненного индекса)

### 2. rag_service

Вэб-сервис на Flask. Инициализация индекса перед первым запросом.

#### GET /health

- если индекс проинициализирован, то сервис возвращает статус ok, 200
- иначе сервис возвращает статус not_ready, 503

#### POST /search

Обязательные параметры: question.  
Опциональные параметры: top_k.

- если параметра question нет, то сервис возвращает `{"error": "Missing question"}, 400`
- если поиск по индексу завершается ошибкой, то сервис возвращает `{"error": "Search failed"}, 500`
- если происходит другая ошибка, то сервис возвращает `{"error": "Internal server error"}, 500`
- если поиск завершается успехом, то сервис возвращает `200` и:
  ```yaml
  answer:
    type: string
    description: ответ по RAG
  
  chunk_title_list:
    type: array
    items:
      type: string
    description: заголовки топ чанков
  
  chunk_texts:
    type: array
    items:
      type: string
    description: тексты топ чанков
  
  question:
    type: string
    description: значение параметра question
  
  top_k:
    type: integer
    description: значение параметра top_k
  ```
