# flake8: noqa: D205, D400 - module-level utility descriptions kept concise
"""Helpers for composing note_style system prompts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


_BASE_SYSTEM_PROMPT = (
    "You extract structured style from a customer's free-text note. Return JSON ONLY with "
    "schema: {\"tone\": <string>, \"context_hints\": {\"timeframe\": {\"month\": <string|null>, "
    "\"relative\": <string|null>}, \"topic\": <string>, \"entities\": {\"creditor\": <string|null>, "
    "\"amount\": <number|null>}}, \"emphasis\": [<string>...], \"confidence\": <float>, \"risk_flags\": [<string>...]}. "
    "Rules: base decisions on note_text; treat all other fields as orientation only; keep values short; lists ≤6; add "
    "[\"unsupported_claim\"] if the note asserts a legal claim with no supporting docs; for short/ambiguous notes set "
    "confidence ≤0.5; respond with JSON only."
)

_CONTEXT_HINT_PREFIX = "Context hints (orientation only): "
_MAX_HINTS = 6
_MAX_HINT_LENGTH = 120


def build_base_system_prompt() -> str:
    """Return the shared base system prompt text."""

    return _BASE_SYSTEM_PROMPT


def build_context_hint_text(
    account_context: Mapping[str, Any] | None,
    bureaus_summary: Mapping[str, Any] | None,
) -> str:
    """Return a short human-readable hint string for system prompts.

    The hints are intended to orient the model without restating the full context
    payload. Values are trimmed aggressively to keep the prompt compact.
    """

    hints = list(_iter_account_context_hints(account_context))
    hints.extend(_iter_bureau_hints(bureaus_summary))

    normalized: list[str] = []
    for hint in hints:
        clean = _normalize_text(hint)
        if not clean:
            continue
        if clean in normalized:
            continue
        if len(clean) > _MAX_HINT_LENGTH:
            clean = clean[: _MAX_HINT_LENGTH - 1].rstrip() + "…"
        normalized.append(clean)
        if len(normalized) >= _MAX_HINTS:
            break

    if not normalized:
        return ""

    return _CONTEXT_HINT_PREFIX + "; ".join(normalized)


def _iter_account_context_hints(
    account_context: Mapping[str, Any] | None,
) -> Iterable[str]:
    if not isinstance(account_context, Mapping):
        return []

    hints: list[str] = []

    reported_creditor = _normalize_text(account_context.get("reported_creditor"))
    if reported_creditor:
        hints.append(f"creditor={reported_creditor}")

    primary_issue = _normalize_text(account_context.get("primary_issue"))
    if primary_issue:
        hints.append(f"issue={primary_issue}")

    account_tail = _normalize_text(account_context.get("account_tail"))
    if account_tail:
        hints.append(f"acct_tail=…{account_tail}")

    tags = account_context.get("tags") if isinstance(account_context.get("tags"), Mapping) else None
    if isinstance(tags, Mapping):
        issues = tags.get("issues")
        if isinstance(issues, Iterable) and not isinstance(issues, (str, bytes, bytearray)):
            for issue in issues:
                text = _normalize_text(issue)
                if text and text != primary_issue:
                    hints.append(f"tag={text}")
                    break

    return hints


def _iter_bureau_hints(
    bureaus_summary: Mapping[str, Any] | None,
) -> Iterable[str]:
    if not isinstance(bureaus_summary, Mapping):
        return []

    majority = bureaus_summary.get("majority_values")
    if not isinstance(majority, Mapping):
        return []

    hints: list[str] = []

    account_type = _normalize_text(majority.get("account_type"))
    if account_type:
        hints.append(f"type={account_type}")

    account_status = _normalize_text(majority.get("account_status"))
    if account_status:
        hints.append(f"status={account_status}")

    payment_status = _normalize_text(majority.get("payment_status"))
    if payment_status and payment_status != account_status:
        hints.append(f"payment={payment_status}")

    balance = _normalize_text(majority.get("balance_owed"))
    if balance:
        hints.append(f"balance={balance}")

    past_due = _normalize_text(majority.get("past_due_amount"))
    if past_due and past_due != balance:
        hints.append(f"past_due={past_due}")

    return hints


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


__all__ = [
    "build_base_system_prompt",
    "build_context_hint_text",
]
