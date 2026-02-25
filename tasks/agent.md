
# Цель

Написать функцию `agent()`, которая будет принимать на вход `message_history` и возвращать ответ агента в виде строки (+ экстра инфо).


## Вариант 1

Пример message_history:
```python
message_history = [
    {
        "role": "user",
        "content": "what is haiku"
    }
]
```

Пример ответа функции agent:
```python
example_response = {
    "answer": "Haiku — это традиционное японское трёхстишие со строгой структурой 5-7-5 слогов, обычно с отсылкой к природе.",
    "token_usage": {
        "prompt_tokens": 18,
        "completion_tokens": 24,
        "total_tokens": 42
    }
}
```

