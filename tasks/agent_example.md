# Специфика реализации

Здесь представлены детали реализации агента (генерация данных, апи и прочее), описанного в [agent.md](agent.md)

## Дополнение к тестам

Тесты api 1-4 в директории src_learn/api:
```
test_api_get_students               # тест api 1
test_api_get_avg_score              # тест api 2
test_api_get_get_avg_overall_score  # тест api 3
test_api_get_search_rag             # тест api 4
```

Тесты llm api в директории src_learn/llm:
```
test_llm  # тест chat_completions, embeddings
```


## Метки pytest

- api - тесты api 1-4
- llm - тесты llm api (chat_completions, embeddings)
- unit - тесты без вызовов llm api (локально)


## Подробно про запросы на LLM API

### 1. Запрос на генерацию /chat/completions

Отдельная функция: `post_chat_completions(payload: dict, **kwargs) -> dict`

- `payload` обязательно содержит `messages`, может содержать доп. параметры (напр., `temperature`, `max_tokens`)
- `kwargs` могут содержать `verbose` (bool, False by default)
- если `payload` не содержит `model`, то устанавливается дефолтное значение `config.default_model` 
- если `verbose=True`, то в лог пишется `payload` (до запроса) и `response` (после успешного запроса) на уровне debug
- если при запросе происходит ошибка, то в лог пишется стектрейс ошибки на уровне error/critical и возвращается словарь вида `{"error": "<exception text>"}`
- если запрос проходит успешно, возвращается стандартный ответ вида [result_chatcomp.json](../meta/gigachat/result_chatcomp.json)

### 2. Запрос на эмбеддинги /embeddings

Отдельная функция: `post_embeddings(payload: dict, **kwargs) -> dict`

- `payload` обязательно содержит `input`
- `kwargs` могут содержать `verbose` (bool, False by default)
- если `payload` не содержит `model`, то устанавливается дефолтное значение `config.default_embedding_model` 
- если `verbose=True`, то в лог пишется `payload` (до запроса) и `response` (после успешного запроса) на уровне debug
- если при запросе происходит ошибка, то в лог пишется стектрейс ошибки на уровне error/critical и возвращается словарь вида `{"error": "< exception text >"}`
- если запрос проходит успешно, возвращается стандартный ответ вида [result_embeddings.json](../meta/gigachat/result_embeddings.json)








## Инструменты

### top_students

Цель:

- достать студентов, записанных на конкретный предмет
- посчитать их оценки за экзамен
- вывести их вместе с оценками, по которым они отсортированы

Параметры:

- required: название предмета
- optional: топ-k (default 3)

Пример 1:
    user: "лучшие студенты по предмету ml"
    agent: "Avery (80.4), John (76.9), Ethan (76.5)"

Пример 2:
    user: "топ лучших по метоптам"
    agent: "Harper (78.6), Evelyn (77.6), Mia (76.8)"

### hardest_questions

Цель:

- выбрать предмет
- посмотреть на студентов, сдававших этот предмет
- посчитать по каждому вопросу экзамена среднюю оценку студента
- вывести самые сложные вопросы (id) вместе со средними оценками

Параметры:

- required: название предмета
- optional: топ-k (default 3)

Пример 1:
    user: "самые сложные вопросы по машинному обучению"
    agent: "[{"question_id": 7, "avg_grade": 69.1}, {"question_id": 9, "avg_grade": 71.2}, {"question_id": 10, "avg_grade": 71.9}]"

### answer_question

Цель:

- найти ответ в базе знаний (некоторый факт в неструктурированных текстовых данных)

Параметры:

- required: вопрос

WIP THOUGHTS

## hardest_xxx

hardest_questions : самые сложные домашки? тесты?
hardest_tests : ...

## rag

программа, имена преподавателей ...

список учебников, программа курса

нужно чтобы агент мог спутать тулы!! преподаватели, предметы... в раг положить распределение студентов по гурппам (спут с распределением по успеваемости..) книги, препы

мультихоп: у какого преподавателя самый сложный экзамен

+1: сходить за содержимым вопросов

можно вобратную сторону: сначалк искать апреподаваетял в раге, затем идти в БД

1. Supports Complex Queries

Example:
Which lecturer teaches the hardest course?
Flow:
1 Hardest questions per subject
2 Find subject with lowest averages
3 RAG → lecturer
Very agent-like.

1. Enables Multi-hop Reasoning ⭐⭐⭐

Example:
Which literature should I read to improve ML exam results?
Flow:
1 hardest_questions ML
2 rag ML booklet
3 literature section
Perfect test scenario.

Category A — Pure RAG
Basic
"Who teaches Machine Learning?"
"What literature is used in Optimization Theory?"
"What topics are covered in Probability Theory?"
"Describe the ML course"
Semantic
"Which course covers neural networks?"
"Where do students learn gradient descent?"
"Which subject includes stochastic processes?"
Better tests retrieval.