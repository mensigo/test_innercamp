"""Pytest hooks and env: duplicate loguru DEBUG to stdout when running tests."""

import os

os.environ.setdefault('LOG_DEBUG_STDOUT', '1')
