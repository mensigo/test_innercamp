## Пример реализации

### 1. count_stats

split_into_syllables_simple - подсчет логов в русском слове
count_syllables_and_words - подсчет статистики по тексту (число слогов, слов)

### 2. haiku_service

Вэб-сервис на Flask.

#### GET /health

- сервис возвращает статус ok, 200
- формат ответа:
  ```yaml
  status:
    type: string
    description: ok или not_ready
  ```

#### POST /generate_haiku

Обязательные параметры: theme.

- если параметра theme нет, то сервис возвращает `{"error": "Missing theme"}, 400`
- если запрос на генерацию завершается ошибкой, то сервис возвращает `{"error": "Generation failed"}, 500`
- если происходит другая ошибка, то сервис возвращает `{"error": "Internal server error"}, 500`
- если запрос завершается успехом, то сервис возвращает `200` и:
  ```yaml
  haiku_text:
    type: string
    description: текст хайку
  
  syllables_per_line:
    type: array
    items:
      type: integer
    description: число слогов по строкам хайку
  
  total_words:
    type: integer
    description: общее число слов в хайку
  
  theme:
    type: string
    description: значение параметра theme
  ```

