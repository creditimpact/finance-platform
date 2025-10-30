from backend.core.logic.report_analysis import account_merge


def _make_cfg(*, threshold: int = 999, triggers=None) -> account_merge.MergeCfg:
    return account_merge.MergeCfg(
        points={},
        weights={},
        thresholds={"AI_THRESHOLD": threshold},
        triggers=triggers or {},
        tolerances={},
    )


def test_should_build_pack_hard_gate_uses_normalized_level() -> None:
    normalized_level = account_merge._normalized_account_level(
        "none", "exact_or_known_match", "none"
    )
    level_value = account_merge._sanitize_acct_level("none")
    level_value = account_merge._sanitize_acct_level(normalized_level or level_value)

    allowed, reason = account_merge._should_build_pack(
        cfg=_make_cfg(),
        total_score=0,
        dates_all_equal=False,
        level_value=level_value,
    )

    assert allowed
    assert reason == "hard_acctnum"


def test_should_build_pack_respects_hard_toggle_off() -> None:
    allowed, reason = account_merge._should_build_pack(
        cfg=_make_cfg(triggers={"MERGE_AI_ON_HARD_ACCTNUM": False}),
        total_score=0,
        dates_all_equal=False,
        level_value="exact_or_known_match",
    )

    assert not allowed
    assert reason == "below_gate"


def test_should_build_pack_allows_dates_toggle() -> None:
    allowed, reason = account_merge._should_build_pack(
        cfg=_make_cfg(),
        total_score=0,
        dates_all_equal=True,
        level_value="none",
    )

    assert allowed
    assert reason == "dates_all"


def test_should_build_pack_allows_threshold_even_when_triggers_off() -> None:
    allowed, reason = account_merge._should_build_pack(
        cfg=_make_cfg(
            threshold=27,
            triggers={
                "MERGE_AI_ON_HARD_ACCTNUM": False,
                "MERGE_AI_ON_ALL_DATES": False,
            },
        ),
        total_score=30,
        dates_all_equal=False,
        level_value="none",
    )

    assert allowed
    assert reason == "over_threshold"


def test_normalized_account_level_prefers_level_value() -> None:
    assert (
        account_merge._normalized_account_level(
            "none", "exact_or_known_match", "none"
        )
        == "exact_or_known_match"
    )


def test_normalized_account_level_falls_back_to_acct_level() -> None:
    assert (
        account_merge._normalized_account_level(
            "exact_or_known_match", "none", "none"
        )
        == "exact_or_known_match"
    )


def test_normalized_account_level_uses_gate_level_last() -> None:
    assert (
        account_merge._normalized_account_level("none", "none", "exact_or_known_match")
        == "exact_or_known_match"
    )
