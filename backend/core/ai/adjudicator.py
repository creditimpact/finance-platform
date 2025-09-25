"""AI adjudicator client for calling OpenAI's chat completion API."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable

import httpx


_SYSTEM_PROMPT = """You are a meticulous adjudicator for credit-report account pairing.
Decide if two account entries (A,B) refer to the SAME underlying account.

Decisions allowed (exact strings):
- merge                   : same account AND same debt (safe merge)
- same_account            : same account; debt unknown/unclear
- same_account_debt_different : same account; debt conflicts
- same_debt               : same obligation/story; account unknown/unclear
- same_debt_account_different : same obligation; KNOWN different tradelines
- different               : different accounts/obligations

Always output JSON:
{
  "decision": "<one of above>",
  "reason": "short natural language",
  "flags": {"account_match": true|false|"unknown", "debt_match": true|false|"unknown"}
}

Rules:
- If normalized account numbers match (exact or last4 corroborated by lender+dates), set flags.account_match=true.
- If balances/high_balance/past_due + timing align within tolerance → flags.debt_match=true; if they conflict → false.
- If account_match=true AND debt_match=true → decision MUST be "merge".
- If account_match=true AND debt_match=false → "same_account_debt_different".
- If account_match=false AND debt_match=true:
    - If you have evidence accounts are different → "same_debt_account_different";
    - else (unknown account identity) → "same_debt".
- If account_match=true and debt_match is "unknown" → "same_account".
- If debt_match=true and account_match is "unknown" → "same_debt".
- Otherwise → "different".

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


class AdjudicatorError(ValueError):
    """Raised when the AI adjudicator response is malformed."""


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
