import copy
from typing import Callable, Mapping, Tuple

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


@pytest.fixture()
def make_points_accounts() -> Callable[
    ..., Tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]
]:
    base_fields = {
        "account_number": "1234567890",
        "date_opened": "2021-03-15",
        "account_type": "installment",
        "account_status": "open",
        "balance_owed": "100",
        "last_payment": "2023-05-01",
    }

    def _factory(
        *,
        left_override: Mapping[str, object] | None = None,
        right_override: Mapping[str, object] | None = None,
    ) -> Tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
        left_payload = dict(base_fields)
        right_payload = dict(base_fields)
        if left_override:
            left_payload.update(left_override)
        if right_override:
            right_payload.update(right_override)
        return (
            _make_bureaus(transunion=left_payload),
            _make_bureaus(experian=right_payload),
        )

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


def test_points_mode_threshold_progression(
    make_points_cfg, make_points_accounts
) -> None:
    cfg = make_points_cfg()

    low_match_left, low_match_right = make_points_accounts(
        right_override={"balance_owed": "250"}
    )

    low_match_score = account_merge.score_pair_0_100(
        low_match_left, low_match_right, cfg
    )

    assert "last_payment" not in cfg.fields
    assert low_match_score["score_points"] == pytest.approx(3.0)
    assert low_match_score["decision"] == "ai"
    assert "points:ai" in low_match_score["triggers"]
    assert "points:direct" not in low_match_score["triggers"]
    assert low_match_score["field_contributions"]["account_number"] == pytest.approx(1.0)
    assert low_match_score["field_contributions"]["date_opened"] == pytest.approx(1.0)
    assert low_match_score["field_contributions"]["account_type"] == pytest.approx(0.5)
    assert low_match_score["field_contributions"]["account_status"] == pytest.approx(0.5)
    assert low_match_score["field_contributions"]["balance_owed"] == pytest.approx(0.0)
    assert "last_payment" not in low_match_score["fields_evaluated"]
    assert "last_payment" not in low_match_score["field_contributions"]

    direct_left, direct_right = make_points_accounts()
    direct_score = account_merge.score_pair_0_100(direct_left, direct_right, cfg)

    assert direct_score["score_points"] == pytest.approx(6.0)
    assert direct_score["decision"] == "auto"
    assert "points:direct" in direct_score["triggers"]
    assert "points:ai" not in direct_score["triggers"]
    assert direct_score["field_contributions"]["balance_owed"] == pytest.approx(3.0)


def test_points_mode_legacy_field_contributes_zero(
    make_points_cfg, make_points_accounts
) -> None:
    cfg = make_points_cfg()
    assert "last_payment" not in cfg.fields

    with_last_left, with_last_right = make_points_accounts()
    with_last = account_merge.score_pair_0_100(with_last_left, with_last_right, cfg)

    without_last_left, without_last_right = make_points_accounts()
    without_last_left["transunion"].pop("last_payment", None)
    without_last_right["experian"].pop("last_payment", None)
    without_last = account_merge.score_pair_0_100(
        without_last_left, without_last_right, cfg
    )

    assert with_last["score_points"] == pytest.approx(without_last["score_points"])
    assert "last_payment" not in with_last["field_contributions"]
    assert "last_payment" not in with_last["fields_evaluated"]
