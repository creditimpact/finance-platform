"""AI adjudicator client for calling OpenAI's chat completion API."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any, Dict, Iterable

import httpx


_SYSTEM_PROMPT = """You are a meticulous adjudicator for credit-report account pairing.
Decide if two account entries (A,B) refer to the SAME underlying account.

Allowed decisions (exact strings, choose one):
- same_account_same_debt        # accounts align and refer to the same debt
- same_account_diff_debt        # same account, but debt details clearly differ
- same_account_debt_unknown     # same account, debt status cannot be confirmed
- same_debt_diff_account        # same debt, but reported under a different account
- same_debt_account_unknown     # same debt, account identity cannot be confirmed
- different                     # neither the account nor the debt matches
Legacy labels (merge, same_debt, same_debt_account_diff, same_account,
same_account_debt_diff, different) may appear in reference material, but
you MUST respond using only the six decisions above.

Always output strict JSON matching the contract below (no prose around it):
{
  "decision": "one of the six allowed decisions above",
  "flags": {"account_match": true|false|"unknown", "debt_match": true|false|"unknown"},
  "reason": "short natural language"
}

Decision guidance:
- Flags.account_match=true when normalized account numbers (exact or last4 corroborated by lender+dates) align.
- Flags.debt_match=true when balances/high_balance/past_due + timing align within tolerance; false when they conflict.
- If account_match=true and debt_match=true → same_account_same_debt.
- If account_match=true and debt_match=false → same_account_diff_debt.
- If account_match=false and debt_match=true → same_debt_diff_account when tradelines clearly differ; otherwise lean conservative.
- If account_match=true and debt_match="unknown" → same_account_debt_unknown.
- If debt_match=true and account_match="unknown" → same_debt_account_unknown.
- If either flag is "unknown" be conservative and avoid *_diff decisions unless evidence is explicit.
- If both flags are false → different.

Consider:
• High-precision cues: account-number (last4/exact), balance owed equality within tolerances, date alignments.
• Lender names/brands and free-text descriptors from the raw “context” lines.
• The numeric 0–100 match summary as a hint, but override if raw context contradicts it.
Be conservative: if critical fields conflict without plausible explanation → "different".
Do NOT mention these rules in the output."""

_ALLOWED_USER_PAYLOAD_KEYS: Iterable[str] = (
    "sid",
    "pair",
    "numeric_match_summary",
    "tolerances_hint",
    "ids",
    "context",
)


ALLOWED_DECISIONS: set[str] = {
    "same_account_same_debt",
    "same_account_diff_debt",
    "same_account_debt_unknown",
    "same_debt_diff_account",
    "same_debt_account_unknown",
    "different",
}

ALLOWED_FLAGS_ACCOUNT: set[str] = {"true", "false", "unknown"}
ALLOWED_FLAGS_DEBT: set[str] = {"true", "false", "unknown"}


class AdjudicatorError(ValueError):
    """Raised when the AI adjudicator response is malformed."""


def _normalize_match_flag(value: object, *, field: str) -> bool | str:
    """Return a normalized boolean/"unknown" flag from ``value``."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        allowed_values = (
            ALLOWED_FLAGS_ACCOUNT if field == "account_match" else ALLOWED_FLAGS_DEBT
        )
        if lowered not in allowed_values:
            raise AdjudicatorError(
                f"AI adjudicator flags must set {field} to true, false, or \"unknown\""
            )
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "unknown":
            return "unknown"
    raise AdjudicatorError(
        f"AI adjudicator flags must set {field} to true, false, or \"unknown\""
    )


def _decision_for_flags(
    account_flag: bool | str,
    debt_flag: bool | str,
    *,
    requested: str,
) -> str:
    """Return the contract-compliant decision for the provided flags."""

    normalized_requested = requested.strip().lower()
    legacy_map = {
        "merge": "same_account_same_debt",
        "same_account": "same_account_debt_unknown",
        "same_account_debt_different": "same_account_diff_debt",
        "same_account_debt_diff": "same_account_diff_debt",
        "same_debt": "same_debt_account_unknown",
        "same_debt_account_different": "same_debt_diff_account",
        "same_debt_account_diff": "same_debt_diff_account",
    }
    requested_normalized = legacy_map.get(normalized_requested, normalized_requested)

    if account_flag is True and debt_flag is True:
        return "same_account_same_debt"
    if account_flag is True and debt_flag is False:
        return "same_account_diff_debt"
    if account_flag is False and debt_flag is True:
        return "same_debt_diff_account"
    if account_flag is True and debt_flag == "unknown":
        return "same_account_debt_unknown"
    if account_flag == "unknown" and debt_flag is True:
        return "same_debt_account_unknown"
    if account_flag is False and debt_flag == "unknown":
        return "different"
    if account_flag == "unknown" and debt_flag is False:
        return "different"
    if account_flag == "unknown" and debt_flag == "unknown":
        return "different"
    if account_flag is False and debt_flag is False:
        return "different"
    if requested_normalized in ALLOWED_DECISIONS:
        return requested_normalized
    return "different"


def _normalize_and_validate_decision(
    resp: Mapping[str, Any]
) -> tuple[Dict[str, Any], bool]:
    """Return the normalized decision payload and whether normalization occurred."""

    if not isinstance(resp, Mapping):
        raise AdjudicatorError("AI adjudicator response payload must be an object")

    decision_raw = resp.get("decision")
    if not isinstance(decision_raw, str) or not decision_raw.strip():
        raise AdjudicatorError("AI adjudicator decision must be a non-empty string")
    decision_value = decision_raw.strip().lower()

    reason_raw = resp.get("reason")
    if not isinstance(reason_raw, str) or not reason_raw.strip():
        raise AdjudicatorError("AI adjudicator response must include a reason string")
    reason_value = reason_raw.strip()

    flags_raw = resp.get("flags")
    if not isinstance(flags_raw, Mapping):
        raise AdjudicatorError(
            "AI adjudicator response must include flags.account_match/debt_match"
        )

    account_flag = _normalize_match_flag(flags_raw.get("account_match"), field="account_match")
    debt_flag = _normalize_match_flag(flags_raw.get("debt_match"), field="debt_match")

    normalized_decision = _decision_for_flags(account_flag, debt_flag, requested=decision_value)
    if normalized_decision not in ALLOWED_DECISIONS:
        raise AdjudicatorError("AI adjudicator decision was outside the allowed set")

    normalized_flags = {"account_match": account_flag, "debt_match": debt_flag}

    normalized = decision_value not in ALLOWED_DECISIONS or decision_value != normalized_decision

    normalized_payload: Dict[str, Any] = dict(resp)
    normalized_payload["decision"] = normalized_decision
    normalized_payload["reason"] = reason_value
    normalized_payload["flags"] = normalized_flags
    if normalized:
        normalized_payload["normalized"] = True
    else:
        normalized_payload.pop("normalized", None)

    return normalized_payload, normalized


def _coerce_positive_int(value: str | None, *, default: int) -> int:
    """Return a positive integer parsed from ``value`` or ``default``."""

    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _prepare_user_payload(pack: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the allowed keys from ``pack`` for the user message."""

    return {key: pack[key] for key in _ALLOWED_USER_PAYLOAD_KEYS if key in pack}


def decide_merge_or_different(pack: dict, *, timeout: int) -> dict:
    """Return the adjudicator response with decision, reason, and flags.

    May raise transport/HTTP errors; caller handles retries and ai_error tags.
    """

    base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for adjudicator calls")

    model = os.getenv("AI_MODEL")
    if not model:
        raise RuntimeError("AI_MODEL must be set for adjudicator calls")

    request_timeout = _coerce_positive_int(os.getenv("AI_REQUEST_TIMEOUT"), default=timeout)

    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    project_id = (os.getenv("OPENAI_PROJECT_ID") or "").strip()
    if api_key.startswith("sk-proj-"):
        if not project_id:
            raise RuntimeError(
                "OPENAI_PROJECT_ID must be set when using project-scoped OpenAI API keys"
            )
        headers["OpenAI-Project"] = project_id

    org_id = os.getenv("OPENAI_ORG_ID")
    if org_id:
        headers["OpenAI-Organization"] = org_id

    user_payload = _prepare_user_payload(pack)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload)},
    ]

    request_body = {
        "model": model,
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "messages": messages,
    }

    url = f"{base_url}/chat/completions"
    response = httpx.post(url, headers=headers, json=request_body, timeout=request_timeout)
    response.raise_for_status()
    data = response.json()

    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AdjudicatorError("Unexpected response structure from AI adjudicator") from exc

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AdjudicatorError("AI adjudicator response was not valid JSON") from exc

    decision = parsed.get("decision")
    reason = parsed.get("reason")
    if not isinstance(decision, str) or not decision.strip():
        raise AdjudicatorError("AI adjudicator decision must be a non-empty string")
    if not isinstance(reason, str) or not reason.strip():
        raise AdjudicatorError("AI adjudicator response must include a reason string")

    flags = parsed.get("flags")
    if flags is not None and not isinstance(flags, dict):
        raise AdjudicatorError("AI adjudicator flags must be an object when provided")

    if flags is None:
        return {"decision": decision, "reason": reason}

    result = dict(parsed)
    result["decision"] = decision
    result["reason"] = reason
    result["flags"] = dict(flags)
    return result
