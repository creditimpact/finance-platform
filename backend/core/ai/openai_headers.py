"""Compatibility wrapper for legacy imports."""

from __future__ import annotations

from .openai_auth import build_openai_headers

__all__ = ["build_openai_headers"]
