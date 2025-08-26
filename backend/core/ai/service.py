"""Lightweight LLM service stubs used for tests."""

from __future__ import annotations

from typing import Any


def run_llm_prompt(
    system: str, user: Any, *, temperature: float = 0.0, timeout: int | None = None
) -> str:
    """Stubbed LLM caller.

    In production this would call an AI service.  Tests patch this function to
    return deterministic JSON strings or to raise errors.  The default
    implementation simply raises ``NotImplementedError`` to avoid accidental
    network calls.
    """

    raise NotImplementedError("run_llm_prompt is not implemented")


__all__ = ["run_llm_prompt"]
