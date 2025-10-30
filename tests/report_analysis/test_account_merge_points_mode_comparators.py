import copy

import pytest

from backend.core.logic.report_analysis import account_merge


def _make_bureaus(**kwargs: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    base = {"transunion": {}, "experian": {}, "equifax": {}}
    for bureau, values in kwargs.items():
        base[bureau] = values
    return base


def _build_points_cfg() -> account_merge.MergeCfg:
    weights = {
        "account_number": 1.0,
        "date_opened": 1.0,
        "balance_owed": 3.0,
        "account_type": 0.5,
        "account_status": 0.5,
        "history_2y": 1.0,
        "history_7y": 1.0,
    }
    tolerances = {
        "MERGE_TOL_BALANCE_ABS": 0.0,
        "MERGE_TOL_BALANCE_RATIO": 0.0,
        "MERGE_TOL_DATE_DAYS": 0,
        "MERGE_ACCOUNTNUMBER_MATCH_MINLEN": 0,
        "MERGE_HISTORY_SIMILARITY_THRESHOLD": 1.0,
    }
    cfg = account_merge.MergeCfg(
        points={},
        weights=weights,
        thresholds={},
        triggers={"MERGE_AI_ON_HARD_ACCTNUM": True},
        tolerances=tolerances,
        fields=account_merge._POINTS_MODE_FIELD_ALLOWLIST,
        overrides={},
        allowlist_enforce=True,
        allowlist_fields=account_merge._POINTS_MODE_FIELD_ALLOWLIST,
    )
    setattr(cfg, "points_mode", True)
    setattr(cfg, "ai_points_threshold", 3.0)
    setattr(cfg, "direct_points_threshold", 5.0)
    return cfg


@pytest.fixture()
def make_points_cfg():
    def _factory(**tolerance_overrides: object) -> account_merge.MergeCfg:
        cfg = _build_points_cfg()
        cfg.tolerances = copy.deepcopy(cfg.tolerances)
        cfg.tolerances.update(tolerance_overrides)
        return cfg

    return _factory


def test_account_number_respects_min_length(make_points_cfg) -> None:
    cfg = make_points_cfg(MERGE_ACCOUNTNUMBER_MATCH_MINLEN=4)

    bureaus_a = _make_bureaus(transunion={"account_number": "1234"})
    bureaus_b = _make_bureaus(experian={"account_number": "1234"})

    matched = account_merge.score_pair_0_100(bureaus_a, bureaus_b, cfg)
    assert matched["field_matches"]["account_number"] == pytest.approx(1.0)
    assert matched["field_contributions"]["account_number"] == pytest.approx(1.0)

    cfg_short = make_points_cfg(MERGE_ACCOUNTNUMBER_MATCH_MINLEN=5)
    short = account_merge.score_pair_0_100(bureaus_a, bureaus_b, cfg_short)
    assert short["field_matches"]["account_number"] == pytest.approx(0.0)
    assert short["field_contributions"]["account_number"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "days_a,days_b,tolerance,expected",
    [
        ("2020-01-01", "2020-01-06", 5, 1.0),
        ("2020-01-01", "2020-02-01", 5, 0.0),
    ],
)
def test_date_opened_uses_day_tolerance(
    make_points_cfg, days_a: str, days_b: str, tolerance: int, expected: float
) -> None:
    cfg = make_points_cfg(MERGE_TOL_DATE_DAYS=tolerance)

    bureaus_a = _make_bureaus(transunion={"date_opened": days_a})
    bureaus_b = _make_bureaus(experian={"date_opened": days_b})

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, cfg)
    assert scored["field_matches"]["date_opened"] == pytest.approx(expected)
    assert scored["field_contributions"]["date_opened"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "left,right,tol_abs,tol_ratio,expected",
    [
        ("100", "107", 10.0, 0.0, 1.0),
        ("100", "130", 10.0, 0.0, 0.0),
        ("100", "108", 0.0, 0.1, 1.0),
        ("100", "125", 0.0, 0.1, 0.0),
    ],
)
def test_balance_owed_uses_tolerances(
    make_points_cfg,
    left: str,
    right: str,
    tol_abs: float,
    tol_ratio: float,
    expected: float,
) -> None:
    cfg = make_points_cfg(
        MERGE_TOL_BALANCE_ABS=tol_abs,
        MERGE_TOL_BALANCE_RATIO=tol_ratio,
    )

    bureaus_a = _make_bureaus(transunion={"balance_owed": left})
    bureaus_b = _make_bureaus(experian={"balance_owed": right})

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, cfg)
    assert scored["field_matches"]["balance_owed"] == pytest.approx(expected)
    assert scored["field_contributions"]["balance_owed"] == pytest.approx(
        expected * cfg.weights["balance_owed"]
    )


@pytest.mark.parametrize(
    "field,left,right,expected",
    [
        ("account_type", " Installment ", "installment", 1.0),
        ("account_type", "Installment", "revolving", 0.0),
        ("account_status", " OPEN ", "open", 1.0),
        ("account_status", "open", "closed", 0.0),
    ],
)
def test_account_type_and_status_normalized_equality(
    make_points_cfg, field: str, left: str, right: str, expected: float
) -> None:
    cfg = make_points_cfg()

    bureaus_a = _make_bureaus(transunion={field: left})
    bureaus_b = _make_bureaus(experian={field: right})

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, cfg)
    assert scored["field_matches"][field] == pytest.approx(expected)
    assert scored["field_contributions"][field] == pytest.approx(
        expected * cfg.weights[field]
    )


@pytest.mark.parametrize("field", ["history_2y", "history_7y"])
def test_history_fields_use_similarity(make_points_cfg, field: str) -> None:
    cfg = make_points_cfg(MERGE_HISTORY_SIMILARITY_THRESHOLD=0.5)

    left = "OK OK LATE"
    right = "OK LATE"

    bureaus_a = _make_bureaus(transunion={field: left})
    bureaus_b = _make_bureaus(experian={field: right})

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, cfg)
    normalized_left = account_merge.normalize_history_field(left)
    normalized_right = account_merge.normalize_history_field(right)
    expected_similarity = account_merge.history_similarity_score(
        normalized_left, normalized_right
    )

    assert scored["field_matches"][field] == pytest.approx(expected_similarity)
    assert scored["field_contributions"][field] == pytest.approx(
        expected_similarity * cfg.weights[field]
    )
