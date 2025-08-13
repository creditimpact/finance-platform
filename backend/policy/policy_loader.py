"""Utilities for loading and validating the policy rulebook."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft7Validator, ValidationError

_RULEBOOK_PATH = Path(__file__).with_name("rulebook.yaml")
_SCHEMA_PATH = Path(__file__).with_name("rulebook_schema.yaml")

_RULEBOOK_CACHE: Mapping[str, Any] | None = None
_RULEBOOK_VERSION: str | None = None


def load_rulebook() -> Mapping[str, Any]:
    """Load and return the policy rulebook.

    The rulebook is validated against ``rulebook_schema.yaml``. A
    ``ValidationError`` is raised if the rulebook does not conform to the
    schema.
    """

    global _RULEBOOK_CACHE
    if _RULEBOOK_CACHE is not None:
        return _RULEBOOK_CACHE

    data = yaml.safe_load(_RULEBOOK_PATH.read_text(encoding="utf-8"))
    schema = yaml.safe_load(_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft7Validator(schema)
    validator.validate(data)  # May raise ValidationError
    _RULEBOOK_CACHE = data
    return _RULEBOOK_CACHE


def get_rulebook_version() -> str:
    """Return a stable hash representing the current rulebook version."""

    global _RULEBOOK_VERSION
    if _RULEBOOK_VERSION is None:
        _RULEBOOK_VERSION = hashlib.sha256(
            _RULEBOOK_PATH.read_bytes()
        ).hexdigest()
    return _RULEBOOK_VERSION


__all__ = ["load_rulebook", "get_rulebook_version", "ValidationError"]
