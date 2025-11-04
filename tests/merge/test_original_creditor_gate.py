from backend.core.logic.report_analysis import account_merge


def _make_cfg(*, require: bool) -> account_merge.MergeCfg:
    return account_merge.MergeCfg(
        points={},
        weights={},
        thresholds={"AI_THRESHOLD": 3},
        triggers={},
        tolerances={},
        require_original_creditor_for_ai=require,
    )


def test_ai_pack_gate_ignores_requirement_when_disabled() -> None:
    cfg = _make_cfg(require=False)

    allowed, reason = account_merge._ai_pack_gate_allows(cfg, {}, {})

    assert allowed
    assert reason == ""


def test_ai_pack_gate_blocks_missing_original_creditor() -> None:
    cfg = _make_cfg(require=True)
    left = {"transunion": {"account_number": "1234"}}
    right = {"experian": {"original_creditor": ""}}

    allowed, reason = account_merge._ai_pack_gate_allows(cfg, left, right)

    assert not allowed
    assert reason == "missing_original_creditor"


def test_ai_pack_gate_allows_when_original_creditor_present() -> None:
    cfg = _make_cfg(require=True)
    right = {"experian": {"original_creditor": "Acme Servicing"}}

    allowed, reason = account_merge._ai_pack_gate_allows(cfg, {}, right)

    assert allowed
    assert reason == ""


def test_get_merge_cfg_reads_require_original_creditor_flag() -> None:
    cfg = account_merge.get_merge_cfg(
        {"MERGE_REQUIRE_ORIGINAL_CREDITOR_FOR_AI": "1"}
    )

    assert cfg.require_original_creditor_for_ai is True
    assert cfg.MERGE_REQUIRE_ORIGINAL_CREDITOR_FOR_AI is True
