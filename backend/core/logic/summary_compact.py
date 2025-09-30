from typing import Dict, Any, List


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

    def _filter_ms(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {k: d.get(k) for k in KEEP_MS if k in d}
        # force matched_fields booleans only
        if "matched_fields" in out and isinstance(out["matched_fields"], dict):
            out["matched_fields"] = {k: bool(v) for k, v in out["matched_fields"].items()}
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
        out = {k: item.get(k) for k in KEEP_ME if k in item}
        # force matched_fields booleans only
        if "matched_fields" in out and isinstance(out["matched_fields"], dict):
            out["matched_fields"] = {k: bool(v) for k, v in out["matched_fields"].items()}
        return out

    if ms:
        summary["merge_scoring"] = _filter_ms(ms)
    if isinstance(me, list):
        summary["merge_explanations"] = [_filter_me(x) for x in me]

    return summary
