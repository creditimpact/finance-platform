from __future__ import annotations

import json
import os
from typing import Any

from .ai_pack import DEFAULT_MAX_LINES


HIGHLIGHT_KEYS: tuple[str, ...] = (
    "total",
    "triggers",
    "parts",
    "matched_fields",
    "conflicts",
    "acctnum_level",
)

SYSTEM_MESSAGE = (
    "You are an expert credit tradeline merge adjudicator. Review the provided "
    "highlights and short context snippets to decide if the two accounts refer to "
    "the same underlying obligation. Treat the token '--' as missing or "
    "unknown data. Strong triggers represent decisive evidence and take "
    "priority over mid triggers; mid triggers offer supporting evidence but cannot "
    "override conflicts backed by strong signals. Creditor names may appear with "
    "aliases, abbreviations, or formatting differences—treat reasonable variants "
    "as referring to the same source when supported by other evidence. Respond "
    "ONLY with strict JSON following this schema: "
    '{"decision":"merge"|"no_merge","confidence":0..1,"reasons":[...]}. '
    "Do not add commentary or extra keys."
)


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        number = int(str(value))
    except Exception:
        return default
    return number if number > 0 else default


def _limit_context(lines: list[Any], limit: int) -> list[str]:
    if not lines:
        return []
    coerced = [str(item) if item is not None else "" for item in lines]
    if limit <= 0:
        return coerced
    return coerced[:limit]


def _extract_highlights(source: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if not isinstance(source, dict):
        return payload
    for key in HIGHLIGHT_KEYS:
        if key in source:
            payload[key] = source[key]
    return payload


def _build_user_message(pack: dict, max_lines: int) -> str:
    pair = pack.get("pair") or {}
    ids = pack.get("ids") or {}
    highlights = _extract_highlights(pack.get("highlights"))

    context = pack.get("context") or {}
    context_a = _limit_context(list(context.get("a") or []), max_lines)
    context_b = _limit_context(list(context.get("b") or []), max_lines)

    summary = {
        "sid": pack.get("sid", ""),
        "pair": {"a": pair.get("a"), "b": pair.get("b")},
        "account_numbers": {
            "a": ids.get("account_number_a", "--"),
            "b": ids.get("account_number_b", "--"),
        },
        "highlights": highlights,
        "context": {
            "a": context_a,
            "b": context_b,
        },
    }

    return json.dumps(summary, ensure_ascii=False, sort_keys=True)


def build_prompt_from_pack(pack: dict) -> dict[str, str]:
    limits = pack.get("limits") or {}
    default_limit = _coerce_positive_int(os.getenv("AI_PACK_MAX_LINES_PER_SIDE"), DEFAULT_MAX_LINES)
    pack_limit = _coerce_positive_int(limits.get("max_lines_per_side"), default_limit)
    max_lines = min(default_limit, pack_limit)

    user_message = _build_user_message(pack, max_lines)

    return {"system": SYSTEM_MESSAGE, "user": user_message}

