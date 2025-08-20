from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Literal

from backend.analytics.analytics_tracker import (
    check_canary_guardrails,
    emit_counter,
    log_canary_decision,
)

from . import validators


@dataclass
class TemplateDecision:
    template_path: str | None
    required_fields: List[str]
    missing_fields: List[str]
    router_mode: str


def _enabled() -> bool:
    """Return True if the canary router should handle the request.

    Rollback: set environment variable ``ROUTER_CANARY_PERCENT=0`` to disable.
    """

    ceiling = float(os.getenv("ROUTER_RENDER_MS_P95_CEILING", "250"))
    sanitizer_limit = float(os.getenv("ROUTER_SANITIZER_RATE_CAP", "1.0"))
    ai_cap = float(os.getenv("ROUTER_AI_DAILY_BUDGET", "100000"))
    if check_canary_guardrails(ceiling, sanitizer_limit, ai_cap):
        return False

    if "ROUTER_CANARY_PERCENT" not in os.environ and os.getenv(
        "LETTERS_ROUTER_PHASED", ""
    ).lower() in {"1", "true", "yes"}:
        return True

    try:
        percent = int(os.getenv("ROUTER_CANARY_PERCENT", "0"))
    except ValueError:
        percent = 0
    percent = max(0, min(100, percent))
    if percent <= 0:
        return False
    if percent >= 100:
        return True
    return random.randint(1, 100) <= percent


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
        "fraud_dispute": (
            "fraud_dispute_letter_template.html",
            [
                "creditor_name",
                "account_number_masked",
                "bureau",
                "legal_safe_summary",
                "is_identity_theft",
            ],
        ),
        "debt_validation": (
            "debt_validation_letter_template.html",
            [
                "collector_name",
                "account_number_masked",
                "bureau",
                "legal_safe_summary",
                "days_since_first_contact",
            ],
        ),
        "pay_for_delete": (
            "pay_for_delete_letter_template.html",
            [
                "collector_name",
                "account_number_masked",
                "legal_safe_summary",
                "offer_terms",
            ],
        ),
        "mov": (
            "mov_letter_template.html",
            [
                "creditor_name",
                "account_number_masked",
                "legal_safe_summary",
                "cra_last_result",
                "days_since_cra_result",
            ],
        ),
        "personal_info_correction": (
            "personal_info_correction_template.html",
            [
                "client_name",
                "client_address_lines",
                "date_of_birth",
                "ssn_last4",
                "legal_safe_summary",
            ],
        ),
        "cease_and_desist": (
            "cease_and_desist_letter_template.html",
            ["collector_name", "account_number_masked", "legal_safe_summary"],
        ),
        "direct_dispute": (
            "direct_dispute_letter_template.html",
            [
                "creditor_name",
                "account_number_masked",
                "legal_safe_summary",
                "furnisher_address",
            ],
        ),
        "bureau_dispute": (
            "bureau_dispute_letter_template.html",
            [
                "creditor_name",
                "account_number_masked",
                "bureau",
                "legal_safe_summary",
            ],
        ),
        "inquiry_dispute": (
            "inquiry_dispute_letter_template.html",
            [
                "inquiry_creditor_name",
                "account_number_masked",
                "bureau",
                "legal_safe_summary",
                "inquiry_date",
            ],
        ),
        "medical_dispute": (
            "medical_dispute_letter_template.html",
            [
                "creditor_name",
                "account_number_masked",
                "bureau",
                "legal_safe_summary",
                "amount",
                "medical_status",
            ],
        ),
        "paydown_first": (
            "instruction_template.html",
            ["client_name", "date", "accounts_summary", "per_account_actions"],
        ),
    }

    if tag in {"ignore", "paydown_first", "duplicate"}:
        emit_counter(f"router.skipped.{tag}")
    if tag == "ignore":
        return TemplateDecision(
            template_path=None,
            required_fields=[],
            missing_fields=[],
            router_mode="skip",
        )

    if tag == "duplicate":
        if phase == "candidate":
            emit_counter("router.candidate_selected")
            emit_counter("router.candidate_selected.duplicate")
        elif phase in {"final", "finalize"}:
            emit_counter("router.finalized")
            emit_counter("router.finalized.duplicate")
        return TemplateDecision(
            template_path=None,
            required_fields=[],
            missing_fields=[],
            router_mode="memo",
        )

    template_path, required = routes.get(tag, (None, []))

    if not _enabled():
        log_canary_decision("legacy", template_path or "unknown")
        return TemplateDecision(
            template_path=template_path,
            required_fields=required,
            missing_fields=[],
            router_mode="bypass",
        )

    log_canary_decision("canary", template_path or "unknown")

    missing_fields: List[str] = []
    if template_path and tag != "instruction":
        missing_fields = validators.validate_required_fields(
            template_path, ctx, required, validators.CHECKLIST
        )

    if template_path:
        template_name = os.path.basename(template_path)
        # Emit both legacy and tag-specific router metrics for transition
        if phase == "candidate":
            emit_counter("router.candidate_selected")  # deprecated
            if tag:
                emit_counter(f"router.candidate_selected.{tag}")
                emit_counter(
                    f"router.candidate_selected.{tag}.{template_name}"
                )
        elif phase in {"final", "finalize"}:
            emit_counter("router.finalized")  # deprecated
            if tag:
                emit_counter(f"router.finalized.{tag}")

        if missing_fields:
            for field in missing_fields:
                emit_counter(
                    f"router.missing_fields.{tag}.{template_name}.{field}"
                )

    return TemplateDecision(
        template_path=template_path,
        required_fields=required,
        missing_fields=missing_fields,
        router_mode="auto_route",
    )


__all__ = ["TemplateDecision", "select_template"]
