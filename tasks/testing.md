
# Тесты

Проверяется реализацию функции `agent(...)` представлены в директории test_agent.

## Структура

```
test_agent/
├── conftest.py
├── test_agent_full_happy.py    # полный happy-path через все этапы
├── test_agent_integration.py   # интеграционные проверки последовательных шагов
├── test_agent_step1.py         # step1: классификация (classify)
├── test_agent_step2.py         # step2: выбор инструмента (select)
├── test_agent_step3.py         # step3: валидация аргументов (validate)
├── test_agent_step4.py         # step4: исполнение инструмента/haiku-сервиса (execute)
├── test_utils.py               # вспомогательные тесты/фикстуры
```

Файлы step1–step4 покрывают отдельные этапы пайплайна: классификация, выбор инструмента, валидация, выполнение действия. `test_agent_full_happy.py` и `test_agent_integration.py` закрывают сквозные сценарии.