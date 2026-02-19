# Code style
- using python3.11+
- use type hints (simple types like dict, int etc.)
- dont use "-> None" hint if function returns nothing (excess info)
- use simple docstrings (just describe the main purpose of function)

# Spec/implementation moments

For s02_simple_haiku module consider rules:
- if spec tells to log smth to stdout, it means using logger.info with specific prefix
- though spec describes required logging, implemented code may log more details (especially in debug level)