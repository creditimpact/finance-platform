from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Literal


@dataclass
class TemplateDecision:
    template_path: str | None
    required_fields: List[str]
    router_mode: str
    reason: str


def _enabled() -> bool:
    return os.getenv("LETTERS_ROUTER_ENABLED", "").lower() in {"1", "true", "yes"}


def select_template(
    action_tag: str,
    ctx: dict,
    mode: Literal["render", "skip", "auto_route"] = "auto_route",
) -> TemplateDecision:
    """Return the template selection for ``action_tag``.

    When ``LETTERS_ROUTER_ENABLED`` is not set the router simply mirrors the
    previous hard-coded template choices so behavior remains unchanged.
    """

    tag = (action_tag or "").lower()
    routes = {
        "dispute": ("dispute_letter_template.html", []),
        "goodwill": ("goodwill_letter_template.html", []),
        "custom_letter": ("general_letter_template.html", []),
    }

    if tag == "ignore":
        return TemplateDecision(
            template_path=None,
            required_fields=[],
            router_mode="skip",
            reason="ignore action",
        )

    template_path, required = routes.get(tag, (None, []))

    if not _enabled():
        return TemplateDecision(
            template_path=template_path,
            required_fields=required,
            router_mode="bypass",
            reason="router disabled",
        )

    reason = "route matched" if template_path else "no route"
    return TemplateDecision(
        template_path=template_path,
        required_fields=required,
        router_mode=mode,
        reason=reason,
    )


__all__ = ["TemplateDecision", "select_template"]
