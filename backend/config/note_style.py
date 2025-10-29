"""Environment-backed configuration for the note_style stage.

The ``NOTE_STYLE_RESPONSE_MODE`` flag controls how note_style requests ask the
model to return JSON. The supported values are ``"auto"`` (default),
``"tool"``, and ``"json"``. Downstream code keeps the legacy
``"json_object"`` alias working by normalizing it to ``"json"``. ``"auto"``
prefers the tool-calling flow when the tool schema is available, otherwise it
falls back to plain JSON responses.
"""

from __future__ import annotations

import os
from typing import Final

from . import _coerce_positive_int, _warn_default, env_bool

_DEFAULT_RESPONSE_MODE: Final[str] = "auto"
_VALID_RESPONSE_MODES: Final[set[str]] = {
    "auto",
    "tool",
    "json",
    "json_object",
    "prompt_only",
}


def _coerce_response_mode() -> str:
    raw = os.getenv("NOTE_STYLE_RESPONSE_MODE")
    if raw is None:
        return _DEFAULT_RESPONSE_MODE

    candidate = raw.strip().lower()
    if not candidate:
        _warn_default("NOTE_STYLE_RESPONSE_MODE", raw, _DEFAULT_RESPONSE_MODE, "empty")
        return _DEFAULT_RESPONSE_MODE

    if candidate not in _VALID_RESPONSE_MODES:
        _warn_default(
            "NOTE_STYLE_RESPONSE_MODE",
            raw,
            _DEFAULT_RESPONSE_MODE,
            "invalid_choice",
        )
        return _DEFAULT_RESPONSE_MODE

    if candidate == "json_object":
        return "json"

    return candidate


def _coerce_retry_count() -> int:
    return _coerce_positive_int("NOTE_STYLE_RETRY_COUNT", 2, min_value=0)


NOTE_STYLE_RESPONSE_MODE: Final[str] = _coerce_response_mode()
NOTE_STYLE_RETRY_COUNT: Final[int] = _coerce_retry_count()
NOTE_STYLE_STRICT_SCHEMA: Final[bool] = env_bool("NOTE_STYLE_STRICT_SCHEMA", True)


__all__ = [
    "NOTE_STYLE_RESPONSE_MODE",
    "NOTE_STYLE_RETRY_COUNT",
    "NOTE_STYLE_STRICT_SCHEMA",
]

