"""Pytest hooks and env: duplicate loguru DEBUG to stdout when running tests."""

import os

# So that src.logger adds a DEBUG->stdout sink when tests import src
os.environ.setdefault('LOG_DEBUG_STDOUT', '1')
