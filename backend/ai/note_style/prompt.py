# flake8: noqa: D205, D400 - module-level utility descriptions kept concise
"""Helpers for composing note_style system prompts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


_BASE_SYSTEM_PROMPT = """
You analyse customer notes and respond with structured JSON.
You must respond with exactly ONE JSON object and nothing else.
Do not include code fences or explanations.
Use this schema (keys and nesting exactly as written):

{
  "note": string,
  "analysis": {
    "tone": "assertive" | "neutral" | "empathetic" | "polite",
    "context_hints": {
      "timeframe": { "month": int|null, "relative": string|null },
      "topic": string,
      "entities": { "creditor": string|null, "amount": number|null }
    },
    "emphasis": string[],            // e.g., ["paid_in_full","balance_should_be_0"]
    "confidence": number,            // 0..1
    "risk_flags": string[]           // e.g., ["discrepancy_in_balance"]
  }
}
If a field is unknown, return null (not an empty string).

Example output:
{"note":"...", "analysis":{"tone":"assertive","context_hints":{"timeframe":{"month":null,"relative":"2024"},"topic":"auto_loan_payment","entities":{"creditor":"PALISADES FU","amount":5912.0}},"emphasis":["paid_in_full"],"confidence":0.9,"risk_flags":["discrepancy_in_balance"]}}
"""

_TOOL_RESPONSE_INSTRUCTION = (
    "Use the provided tool exactly once and return your full JSON only in tool.arguments. "
    "Do not put text in assistant.content."
)

_JSON_RESPONSE_INSTRUCTION = (
    "Return exactly one valid JSON object that matches the schema."
    " Do not include commentary, prefixes, suffixes, or code fences."
)

_CONTEXT_HINT_PREFIX = "Context hints: "
_MAX_HINTS = 6
_MAX_HINT_LENGTH = 120
_MAJORITY_FIELD_ORDER = (
    ("account_type", "type"),
    ("account_status", "status"),
    ("payment_status", "payment"),
    ("balance_owed", "balance"),
    ("past_due_amount", "past_due"),
    ("date_of_last_activity", "last_activity"),
)
_DISAGREEMENT_FIELD_ORDER = (
    "account_status",
    "payment_status",
    "balance_owed",
    "past_due_amount",
    "date_of_last_activity",
    "date_reported",
)

_BUREAU_FIELD_LABELS = {
    "account_status": "status",
    "payment_status": "payment",
    "balance_owed": "balance",
    "past_due_amount": "past_due",
    "date_of_last_activity": "last_activity",
    "date_reported": "reported",
}


def build_base_system_prompt() -> str:
    """Return the shared base system prompt text."""

    return _BASE_SYSTEM_PROMPT


def build_response_instruction(*, use_tools: bool) -> str:
    """Return the canonical instruction for the active response mode."""

    return _TOOL_RESPONSE_INSTRUCTION if use_tools else _JSON_RESPONSE_INSTRUCTION


def build_context_hint_text(
    account_context: Mapping[str, Any] | None,
    bureaus_summary: Mapping[str, Any] | None,
) -> str:
    """Return a short human-readable hint string for system prompts.

    The hints are intended to orient the model without restating the full context
    payload. Values are trimmed aggressively to keep the prompt compact.
    """

    account_hints = list(_iter_account_context_hints(account_context))
    majority_hints = list(_iter_bureau_hints(bureaus_summary))
    disagreement_hints = list(_iter_disagreement_hints(bureaus_summary))

    hints = _merge_hint_groups(account_hints, majority_hints, disagreement_hints)

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
        hints.append(f"issuer={reported_creditor}")

    primary_issue = _normalize_text(account_context.get("primary_issue"))
    if primary_issue:
        hints.append(f"issue={primary_issue}")

    account_tail = _normalize_text(account_context.get("account_tail"))
    if account_tail:
        hints.append(f"tail=…{account_tail}")

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

    for field, label in _MAJORITY_FIELD_ORDER:
        value = _normalize_text(majority.get(field))
        if not value:
            continue
        hints.append(f"{label}={value}")

    return hints


def _iter_disagreement_hints(
    bureaus_summary: Mapping[str, Any] | None,
) -> Iterable[str]:
    if not isinstance(bureaus_summary, Mapping):
        return []

    disagreements = bureaus_summary.get("disagreements")
    if not isinstance(disagreements, Mapping):
        return []

    hints: list[str] = []

    for field in _DISAGREEMENT_FIELD_ORDER:
        bureau_map = disagreements.get(field)
        if not isinstance(bureau_map, Mapping):
            continue
        entries: list[str] = []
        for bureau, value in sorted(bureau_map.items(), key=lambda item: item[0]):
            clean = _normalize_text(value)
            if not clean:
                continue
            abbr = bureau[:2].upper()
            entries.append(f"{abbr}={clean}")
            if len(entries) >= 3:
                break
        if not entries:
            continue
        label = _BUREAU_FIELD_LABELS.get(field, field)
        hints.append(f"{label}_diff={'/'.join(entries)}")

    return hints


def _merge_hint_groups(*groups: Iterable[str]) -> list[str]:
    sequences = [list(group) for group in groups if group]
    if not sequences:
        return []

    merged: list[str] = []
    max_len = max(len(group) for group in sequences)
    for index in range(max_len):
        for group in sequences:
            if index < len(group):
                merged.append(group[index])

    return merged


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
