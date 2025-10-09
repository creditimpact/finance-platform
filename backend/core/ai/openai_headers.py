"""Shared helpers for building OpenAI request headers."""

from __future__ import annotations

import os


def build_openai_headers(
    api_key: str | None = None,
    project_id: str | None = None,
    content_type: str = "application/json",
) -> dict[str, str]:
    """Construct headers for OpenAI API requests.

    Parameters
    ----------
    api_key:
        The API key to use. When ``None``, the ``OPENAI_API_KEY`` environment
        variable is consulted instead.
    project_id:
        Optional project identifier. Falls back to ``OPENAI_PROJECT_ID``.
    content_type:
        ``Content-Type`` header value to include in the request.
    """

    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    proj = (project_id or os.getenv("OPENAI_PROJECT_ID", "")).strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is empty")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": content_type,
    }
    if key.startswith("sk-proj-"):
        if not proj:
            raise RuntimeError("OPENAI_PROJECT_ID is required for sk-proj-* keys")
        headers["OpenAI-Project"] = proj
    elif proj:
        headers["OpenAI-Project"] = proj

    return headers

