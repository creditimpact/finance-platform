from backend.core.logic.report_analysis.account_merge import get_merge_cfg


def _clear_env(monkeypatch):
    keys = [
        "AI_THRESHOLD",
        "AUTO_MERGE_THRESHOLD",
        "MERGE_AI_ON_BALOWED_EXACT",
        "MERGE_AI_ON_ACCTNUM_LEVEL",
        "MERGE_AI_ON_MID_K",
        "MERGE_AI_ON_ALL_DATES",
        "AMOUNT_TOL_ABS",
        "AMOUNT_TOL_RATIO",
        "LAST_PAYMENT_DAY_TOL",
        "COUNT_ZERO_PAYMENT_MATCH",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_get_merge_cfg_defaults(monkeypatch):
    _clear_env(monkeypatch)

    cfg = get_merge_cfg()

    assert sum(cfg.points.values()) == 96
    assert cfg.thresholds["AI_THRESHOLD"] == 26
    assert cfg.thresholds["AUTO_MERGE_THRESHOLD"] == 70
    assert cfg.triggers["MERGE_AI_ON_BALOWED_EXACT"] is True
    assert cfg.triggers["MERGE_AI_ON_ACCTNUM_LEVEL"] == "last6"
    assert cfg.triggers["MERGE_AI_ON_MID_K"] == 26
    assert cfg.triggers["MERGE_AI_ON_ALL_DATES"] is True
    assert cfg.tolerances["AMOUNT_TOL_ABS"] == 50.0
    assert cfg.tolerances["AMOUNT_TOL_RATIO"] == 0.01
    assert cfg.tolerances["LAST_PAYMENT_DAY_TOL"] == 7
    assert cfg.tolerances["COUNT_ZERO_PAYMENT_MATCH"] == 0


def test_get_merge_cfg_env_overrides():
    overrides = {
        "AI_THRESHOLD": "33",
        "AUTO_MERGE_THRESHOLD": "80",
        "MERGE_AI_ON_BALOWED_EXACT": "0",
        "MERGE_AI_ON_ACCTNUM_LEVEL": "ANY",
        "MERGE_AI_ON_MID_K": "30",
        "MERGE_AI_ON_ALL_DATES": "0",
        "AMOUNT_TOL_ABS": "75.5",
        "AMOUNT_TOL_RATIO": "0.05",
        "LAST_PAYMENT_DAY_TOL": "10",
        "COUNT_ZERO_PAYMENT_MATCH": "1",
    }

    cfg = get_merge_cfg(overrides)

    assert cfg.thresholds["AI_THRESHOLD"] == 33
    assert cfg.thresholds["AUTO_MERGE_THRESHOLD"] == 80
    assert cfg.triggers["MERGE_AI_ON_BALOWED_EXACT"] is False
    assert cfg.triggers["MERGE_AI_ON_ACCTNUM_LEVEL"] == "any"
    assert cfg.triggers["MERGE_AI_ON_MID_K"] == 30
    assert cfg.triggers["MERGE_AI_ON_ALL_DATES"] is False
    assert cfg.tolerances["AMOUNT_TOL_ABS"] == 75.5
    assert cfg.tolerances["AMOUNT_TOL_RATIO"] == 0.05
    assert cfg.tolerances["LAST_PAYMENT_DAY_TOL"] == 10
    assert cfg.tolerances["COUNT_ZERO_PAYMENT_MATCH"] == 1

    assert sum(cfg.points.values()) == 96
