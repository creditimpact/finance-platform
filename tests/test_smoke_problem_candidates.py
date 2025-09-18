from typing import Any, Dict, Optional

from scripts.smoke_problem_candidates import _build_merge_summary


def _make_candidate(
    idx: int,
    *,
    best_index: Optional[int],
    score: Optional[float],
    decision: Optional[str],
    account_decision: str,
    reasons: Optional[Any] = None,
    aux: Optional[Dict[str, Any]] = None,
    parts: Optional[Dict[str, Any]] = None,
):
    merge_tag: dict = {
        "group_id": f"g{idx + 1}",
        "decision": account_decision,
        "best_match": {},
        "parts": parts or {},
    }
    if best_index is not None:
        merge_tag["best_match"] = {
            "account_index": best_index,
            "score": score,
            "decision": decision,
        }
        if reasons:
            if isinstance(reasons, list):
                payload = [dict(item) for item in reasons]
            else:
                payload = dict(reasons)
            merge_tag["best_match"]["reasons"] = payload
            merge_tag["reasons"] = payload
    else:
        merge_tag["best_match"] = {}
    if aux is not None:
        merge_tag["aux"] = dict(aux)
    elif reasons:
        merge_tag["aux"] = {"override_reasons": dict(reasons)}
    return {"merge_tag": merge_tag}


def test_build_merge_summary_includes_override_columns():
    candidates = [
        _make_candidate(
            0,
            best_index=1,
            score=0.42,
            decision="ai",
            account_decision="ai",
            reasons=[{"kind": "acctnum", "level": "last4", "masked_any": True}],
            aux={
                "acctnum_level": "last4",
                "override_reasons": {
                    "acctnum_only_triggers_ai": True,
                    "acctnum_match_level": "last4",
                },
                "override_reason_entries": [
                    {"kind": "acctnum", "level": "last4", "masked_any": True}
                ],
            },
            parts={"acct_num": 0.7},
        ),
        _make_candidate(
            1,
            best_index=0,
            score=0.41,
            decision="ai",
            account_decision="ai",
            reasons={"balance_only_triggers_ai": True},
            aux={
                "acctnum_level": "none",
                "override_reasons": {"balance_only_triggers_ai": True},
            },
            parts={"balowed": 1.0},
        ),
        _make_candidate(
            2,
            best_index=None,
            score=None,
            decision=None,
            account_decision="different",
            aux={"acctnum_level": "none"},
        ),
    ]

    summary = _build_merge_summary(candidates)
    pairs = summary["pairs"]

    assert len(pairs) == 3

    first = pairs[0]
    assert first["decision"] == "ai"
    assert first["acctnum_level"] == "last4"
    assert first["balowed_ok"] is False
    assert any(item.startswith("acctnum(") for item in first["reasons"])

    second = pairs[1]
    assert second["balowed_ok"] is True
    assert "balance_only_triggers_ai" in second["reasons"]

    third = pairs[2]
    assert third["decision"] is None or third["decision"] == "different"
    assert third["reasons"] == []


def test_build_merge_summary_only_ai_filter():
    candidates = [
        _make_candidate(
            0,
            best_index=1,
            score=0.5,
            decision="ai",
            account_decision="ai",
            aux={"acctnum_level": "exact"},
        ),
        _make_candidate(
            1,
            best_index=0,
            score=0.5,
            decision="ai",
            account_decision="ai",
            aux={"acctnum_level": "exact"},
        ),
        _make_candidate(
            2,
            best_index=None,
            score=None,
            decision=None,
            account_decision="different",
            aux={"acctnum_level": "none"},
        ),
    ]

    summary = _build_merge_summary(candidates, only_ai=True)
    pairs = summary["pairs"]

    assert len(pairs) == 2
    assert all(entry["decision"] == "ai" for entry in pairs)
