from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def compact_merge_sections(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mutates a summary dict in-place to keep only compact merge_scoring and merge_explanations.
    Returns the same dict (for chaining).
    """
    ms = summary.get("merge_scoring") or {}
    me = summary.get("merge_explanations") or []

    KEEP_MS = {
        "best_with",
        "score_total",
        "reasons",
        "conflicts",
        "identity_score",
        "debt_score",
        "acctnum_level",
        "matched_fields",
        "acctnum_digits_len_a",
        "acctnum_digits_len_b",
    }

    def _list_of_str(value: Any) -> list[str] | None:
        if isinstance(value, set):
            return sorted((str(v) for v in value))
        if isinstance(value, (list, tuple)):
            return [str(v) for v in value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [str(v) for v in value]
        return None

    def _safe_int(value: Any) -> int | None:
        if isinstance(value, bool):  # bool is int subclass; keep explicit bools elsewhere
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    def _bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        return bool(value)

    def _filter_ms(d: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in KEEP_MS:
            if key not in d:
                continue
            value = d.get(key)
            if key in {
                "best_with",
                "score_total",
                "identity_score",
                "debt_score",
                "acctnum_digits_len_a",
                "acctnum_digits_len_b",
            }:
                iv = _safe_int(value)
                if iv is not None:
                    out[key] = iv
            elif key in {"reasons", "conflicts"}:
                list_value = _list_of_str(value)
                if list_value is not None:
                    out[key] = list_value
            elif key == "acctnum_level":
                if isinstance(value, str):
                    out[key] = value
            elif key == "matched_fields":
                if isinstance(value, Mapping):
                    out[key] = {k: bool(v) for k, v in value.items()}
            else:
                out[key] = value
        return out

    KEEP_ME = {
        "kind",
        "with",
        "decision",
        "total",
        "parts",
        "matched_fields",
        "reasons",
        "conflicts",
        "strong",
        "acctnum_level",
        "acctnum_digits_len_a",
        "acctnum_digits_len_b",
    }

    def _filter_me(item: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in KEEP_ME:
            if key not in item:
                continue
            value = item.get(key)
            if key in {"with", "total", "acctnum_digits_len_a", "acctnum_digits_len_b"}:
                iv = _safe_int(value)
                if iv is not None:
                    out[key] = iv
            elif key == "parts":
                if isinstance(value, Mapping):
                    parts: Dict[str, int] = {}
                    for part_key, part_value in value.items():
                        iv = _safe_int(part_value)
                        if iv is not None:
                            parts[str(part_key)] = iv
                    out[key] = parts
            elif key in {"reasons", "conflicts"}:
                list_value = _list_of_str(value)
                if list_value is not None:
                    out[key] = list_value
            elif key == "matched_fields":
                if isinstance(value, Mapping):
                    out[key] = {k: bool(v) for k, v in value.items()}
            elif key == "strong":
                bool_value = _bool(value)
                if bool_value is not None:
                    out[key] = bool_value
            elif key == "acctnum_level":
                if isinstance(value, str):
                    out[key] = value
            else:
                out[key] = value
        return out

    if isinstance(ms, Mapping):
        summary["merge_scoring"] = _filter_ms(dict(ms))
    if isinstance(me, list):
        summary["merge_explanations"] = [
            _filter_me(dict(item))
            for item in me
            if isinstance(item, Mapping)
        ]

    return summary
