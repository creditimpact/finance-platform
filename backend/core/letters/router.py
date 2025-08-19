from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Literal

from backend.analytics.analytics_tracker import emit_counter
from . import validators


@dataclass
class TemplateDecision:
    template_path: str | None
    required_fields: List[str]
    missing_fields: List[str]
    router_mode: str


def _enabled() -> bool:
    return os.getenv("LETTERS_ROUTER_PHASED", "").lower() in {"1", "true", "yes"}


def select_template(
    action_tag: str,
    ctx: dict,
    phase: Literal["candidate", "final", "finalize"],
) -> TemplateDecision:
    """Return the template selection for ``action_tag``.

    When ``LETTERS_ROUTER_PHASED`` is not set the router simply mirrors the
    previous hard-coded template choices so behavior remains unchanged.
    """

    tag = (action_tag or "").lower()
    routes = {
        "dispute": ("dispute_letter_template.html", ["bureau"]),
        "goodwill": ("goodwill_letter_template.html", ["creditor"]),
        "custom_letter": ("general_letter_template.html", ["recipient"]),
        "instruction": (
            "instruction_template.html",
            ["client_name", "date", "accounts_summary", "per_account_actions"],
        ),
    }

    if tag == "ignore":
        return TemplateDecision(
            template_path=None,
            required_fields=[],
            missing_fields=[],
            router_mode="skip",
        )

    template_path, required = routes.get(tag, (None, []))

    if not _enabled():
        return TemplateDecision(
            template_path=template_path,
            required_fields=required,
            missing_fields=[],
            router_mode="bypass",
        )

    missing_fields: List[str] = []
    if template_path:
        missing_fields = validators.validate_required_fields(
            template_path, ctx, required, validators.CHECKLIST
        )

    if template_path:
        if phase == "candidate":
            emit_counter("router.candidate_selected")
        elif phase in {"final", "finalize"}:
            emit_counter("router.finalized")

    if missing_fields:
        for field in missing_fields:
            emit_counter(f"router.missing_fields.{template_path}.{field}")

    return TemplateDecision(
        template_path=template_path,
        required_fields=required,
        missing_fields=missing_fields,
        router_mode="auto_route",
    )


__all__ = ["TemplateDecision", "select_template"]

