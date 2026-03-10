# Code style
- using python3.11+
- use type hints (simple types like dict, int etc.)
- dont use "-> None" hint if function returns nothing (excess info)
- use simple docstrings (just describe the main purpose of function)

# Commands
- to run `python` or `uv` activate `.venv` first

# Spec/implementation moments

Spec rules:
- if spec tells to log smth to stdout, it means using logger.info with specific prefix
- though spec describes required logging, implemented code may log more details (especially in debug level)

Prompt rules:
- write prompts in russian language

Testing:
- make sure to activate python `.venv` before running any tests
- use `uv run pytest <file>` command to run any tests if needed
- if test includes llm call, use `uv run pytest -n 4`

Agent implementation:

If you are asked to modify agent in src_example/, then follow rules:
- try not to use regex, prefer prompts
- try to extend existing approach smoothly
- modify _build_vector_query strings if needed (just use strings with general info, do not include data from chunks, that can possibly change, i.e. lecturer, schedule, groups etc)

# Project structure

src - template for agent described in tasks/agent.md
test - tests for src

src_example - example agent implementation, according to tasks/agent_example.md
test_example - tests for example agent implementation, according to tasks/agent_example.md