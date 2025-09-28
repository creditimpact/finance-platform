from backend.core.logic.report_analysis import account_merge


def test_should_build_pack_hard_gate_uses_normalized_level() -> None:
    # Simulate a pair where the normalized account-number level indicates a hard match
    # even though the raw gate_level does not.
    acct_aux = {"acctnum_level": "exact_or_known_match"}
    allow_flags = {
        "hard_acct": account_merge._sanitize_acct_level(acct_aux["acctnum_level"]) == "exact_or_known_match",
        "gate_level": "none",
        "dates_all": False,
        "dates": False,
        "total": False,
    }

    cfg = account_merge.MergeCfg(
        points={},
        thresholds={"AI_THRESHOLD": 999},
        triggers={},
        tolerances={},
    )

    assert account_merge._should_build_pack(0, allow_flags, cfg)
