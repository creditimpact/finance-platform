#!/usr/bin/env bash
set -euo pipefail

ruff check tests/test_architecture.py
black --check tests/test_architecture.py
# Run mypy on the typed models package
mypy backend/core/models
python scripts/scan_public_dict_apis.py
pytest -q
