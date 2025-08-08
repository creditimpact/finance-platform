#!/usr/bin/env bash
set -euo pipefail

ruff check tests/test_architecture.py
black --check tests/test_architecture.py
mypy models
pytest -q
