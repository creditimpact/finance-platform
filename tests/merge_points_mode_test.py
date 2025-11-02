"""Regression tests for strict points-mode merge semantics."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

import pytest

from backend.config.merge_config import POINTS_MODE_DEFAULT_WEIGHTS
from backend.core.logic.report_analysis import account_merge


_ALLOWED_SIGNALS = (
    "account_number",
    "date_opened",
    "balance_owed",
    "account_type",
    "account_status",
    "history_2y",
    "history_7y",
)


def _make_bureaus(**kwargs: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    base: Dict[str, Dict[str, Any]] = {"transunion": {}, "experian": {}, "equifax": {}}
    for bureau, values in kwargs.items():
        base[bureau] = dict(values)
    return base


def _build_points_cfg() -> account_merge.MergeCfg:
    weights = dict(POINTS_MODE_DEFAULT_WEIGHTS)
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
        fields=_ALLOWED_SIGNALS,
        overrides={},
        allowlist_enforce=True,
        allowlist_fields=_ALLOWED_SIGNALS,
    )
    setattr(cfg, "points_mode", True)
    setattr(cfg, "ai_points_threshold", 3.0)
    setattr(cfg, "direct_points_threshold", 5.0)
    setattr(cfg, "field_sequence", _ALLOWED_SIGNALS)
    setattr(cfg, "weights_map", dict(weights))
    return cfg


@pytest.fixture()
def points_cfg() -> account_merge.MergeCfg:
    return _build_points_cfg()


def _all_signals_payload(**overrides: Any) -> Dict[str, Any]:
    payload = {
        "account_number": "1234567890",
        "date_opened": "2021-01-01",
        "balance_owed": "10000",
        "account_type": "installment",
        "account_status": "open",
        "history_2y": "OK OK",
        "history_7y": "OK OK",
    }
    payload.update(overrides)
    return payload


def test_balance_mismatch_emits_conflict_and_blocks_direct(points_cfg: account_merge.MergeCfg) -> None:
    left_payload = _all_signals_payload(balance_owed="10123")
    right_payload = _all_signals_payload(balance_owed="10424")

    bureaus_a = _make_bureaus(transunion=left_payload)
    bureaus_b = _make_bureaus(experian=right_payload)

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    assert scored["parts"]["balance_owed"] == pytest.approx(0.0)
    assert "amount_conflict:balance_owed" in scored["conflicts"]
    assert scored["decision"] == "ai"
    assert "points:direct" not in scored["triggers"]
    non_balance_total = sum(
        scored["parts"][field] for field in _ALLOWED_SIGNALS if field != "balance_owed"
    )
    assert scored["total"] == pytest.approx(non_balance_total)


def test_balance_exact_match_any_bureau_unlocks_direct(points_cfg: account_merge.MergeCfg) -> None:
    left_transunion = _all_signals_payload(balance_owed="10000")
    left_experian = _all_signals_payload(balance_owed="20000")
    right_equifax = _all_signals_payload(balance_owed="10000")
    right_experian = _all_signals_payload(balance_owed="30000")

    bureaus_a = _make_bureaus(transunion=left_transunion, experian=left_experian)
    bureaus_b = _make_bureaus(experian=right_experian, equifax=right_equifax)

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    assert scored["parts"]["balance_owed"] == pytest.approx(
        points_cfg.weights["balance_owed"]
    )
    assert "amount_conflict:balance_owed" not in scored["conflicts"]
    assert scored["decision"] == "auto"
    assert "points:direct" in scored["triggers"]
    expected_total = sum(points_cfg.weights[field] for field in _ALLOWED_SIGNALS)
    assert scored["total"] == pytest.approx(expected_total)


def test_balance_exact_match_handles_currency_format(points_cfg: account_merge.MergeCfg) -> None:
    left_payload = _all_signals_payload(balance_owed="$10,000.00")
    right_payload = _all_signals_payload(balance_owed="10000")

    bureaus_a = _make_bureaus(transunion=left_payload)
    bureaus_b = _make_bureaus(experian=right_payload)

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    assert scored["parts"]["balance_owed"] == pytest.approx(points_cfg.weights["balance_owed"])
    assert "amount_conflict:balance_owed" not in scored["conflicts"]
    assert scored["decision"] == "auto"


def test_balance_missing_counts_as_conflict(points_cfg: account_merge.MergeCfg) -> None:
    left_payload = _all_signals_payload(balance_owed="2500")
    right_payload = _all_signals_payload(balance_owed=None)

    bureaus_a = _make_bureaus(transunion=left_payload)
    bureaus_b = _make_bureaus(experian=right_payload)

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    assert scored["parts"]["balance_owed"] == pytest.approx(0.0)
    assert "amount_conflict:balance_owed" in scored["conflicts"]
    assert scored["decision"] == "ai"


def test_balance_close_values_still_conflict(points_cfg: account_merge.MergeCfg) -> None:
    left_payload = _all_signals_payload(balance_owed="10000.00")
    right_payload = _all_signals_payload(balance_owed="10000.01")

    bureaus_a = _make_bureaus(experian=left_payload)
    bureaus_b = _make_bureaus(equifax=right_payload)

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    assert scored["parts"]["balance_owed"] == pytest.approx(0.0)
    assert "amount_conflict:balance_owed" in scored["conflicts"]
    assert scored["decision"] == "ai"


def test_balance_zero_values_do_not_match(points_cfg: account_merge.MergeCfg) -> None:
    left_payload = _all_signals_payload(balance_owed="0")
    right_payload = _all_signals_payload(balance_owed="0.00")

    bureaus_a = _make_bureaus(transunion=left_payload)
    bureaus_b = _make_bureaus(experian=right_payload)

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    assert scored["parts"]["balance_owed"] == pytest.approx(0.0)
    assert "amount_conflict:balance_owed" in scored["conflicts"]
    assert scored["decision"] == "ai"


def test_points_mode_parts_pure_and_allowlisted(points_cfg: account_merge.MergeCfg) -> None:
    bureaus_a = _make_bureaus(transunion=_all_signals_payload())
    bureaus_b = _make_bureaus(experian=_all_signals_payload())

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    assert set(scored["parts"].keys()) == set(_ALLOWED_SIGNALS)
    assert set(scored["field_contributions"].keys()) == set(_ALLOWED_SIGNALS)
    assert tuple(sorted(scored["fields_evaluated"])) == tuple(sorted(_ALLOWED_SIGNALS))
    assert scored["total"] == pytest.approx(sum(scored["parts"].values()))


def test_ai_threshold_never_yields_direct(points_cfg: account_merge.MergeCfg) -> None:
    left_payload = _all_signals_payload(history_2y=None, history_7y=None)
    right_payload = _all_signals_payload(
        date_opened="2019-02-02",
        account_type="revolving",
        account_status="closed",
        history_2y=None,
        history_7y=None,
    )
    left_payload["balance_owed"] = "10000"
    right_payload["balance_owed"] = "10000"

    bureaus_a = _make_bureaus(transunion=left_payload)
    bureaus_b = _make_bureaus(experian=right_payload)

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    assert scored["total"] < points_cfg.direct_points_threshold
    assert scored["total"] >= points_cfg.ai_points_threshold
    assert scored["parts"]["balance_owed"] == pytest.approx(points_cfg.weights["balance_owed"])
    assert scored["parts"]["account_number"] == pytest.approx(points_cfg.weights["account_number"])
    assert scored["decision"] == "ai"
    assert "points:direct" not in scored["triggers"]
    assert "points:ai" in scored["triggers"]
    assert "amount_conflict:balance_owed" not in scored["conflicts"]


def test_points_mode_parts_follow_matched_flags(points_cfg: account_merge.MergeCfg) -> None:
    left_payload = _all_signals_payload(
        account_type="installment",
        account_status="open",
        account_number="111122223333",
    )
    right_payload = _all_signals_payload(
        account_type="revolving",
        account_status="closed",
        account_number="999900001111",
    )
    right_payload["balance_owed"] = left_payload["balance_owed"]

    bureaus_a = _make_bureaus(transunion=left_payload)
    bureaus_b = _make_bureaus(experian=right_payload)

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    account_type_aux = scored["aux"]["account_type"]
    assert account_type_aux["points_mode_matched_bool"] is False
    assert scored["parts"]["account_type"] == pytest.approx(0.0)

    account_number_aux = scored["aux"]["account_number"]
    assert account_number_aux.get("acctnum_level") == "none"
    assert account_number_aux["points_mode_matched_bool"] is False
    assert scored["parts"]["account_number"] == pytest.approx(0.0)

    total_from_parts = sum(scored["parts"].values())
    assert scored["total"] == pytest.approx(total_from_parts)

    aux_payload = account_merge._build_aux_payload(scored["aux"], cfg=points_cfg)
    matched_fields = aux_payload["matched_fields"]
    for field in _ALLOWED_SIGNALS:
        expected = bool(scored["parts"][field] > 0.0)
        assert matched_fields[field] is expected


def test_points_mode_total_not_clamped_to_threshold(
    points_cfg: account_merge.MergeCfg,
) -> None:
    left_payload = _all_signals_payload()
    right_payload = _all_signals_payload(
        account_number="99990000",
        date_opened="2019-12-12",
        balance_owed="25000",
        history_2y="LATE LATE",
        history_7y="LATE LATE",
    )

    bureaus_a = _make_bureaus(transunion=left_payload)
    bureaus_b = _make_bureaus(experian=right_payload)

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    assert scored["parts"]["account_type"] == pytest.approx(
        points_cfg.weights["account_type"]
    )
    assert scored["parts"]["account_status"] == pytest.approx(
        points_cfg.weights["account_status"]
    )
    for field in _ALLOWED_SIGNALS:
        if field in {"account_type", "account_status"}:
            continue
        assert scored["parts"][field] == pytest.approx(0.0)

    expected_total = (
        points_cfg.weights["account_type"] + points_cfg.weights["account_status"]
    )
    assert scored["total"] == pytest.approx(expected_total)
    assert scored["total"] < points_cfg.ai_points_threshold


def test_account_number_points_require_resolved_level(
    monkeypatch: pytest.MonkeyPatch, points_cfg: account_merge.MergeCfg
) -> None:
    def fake_match_account_number_best_pair(*_: Any, **__: Any) -> tuple[bool, Dict[str, Any]]:
        return True, {
            "acctnum_level": "none",
            "matched": True,
            "matched_bool": True,
            "match_score": 1.0,
            "best_pair": ("transunion", "experian"),
            "normalized_values": ("1111", "1111"),
            "raw_values": {"a": "1111", "b": "1111"},
        }

    monkeypatch.setattr(
        account_merge,
        "_match_account_number_best_pair",
        fake_match_account_number_best_pair,
    )

    bureaus_a = _make_bureaus(transunion=_all_signals_payload())
    bureaus_b = _make_bureaus(experian=_all_signals_payload())

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    assert scored["parts"]["account_number"] == pytest.approx(0.0)
    account_aux = scored["aux"]["account_number"]
    assert account_aux["points_mode_acctnum_level"] == "none"
    assert account_aux["points_mode_matched_bool"] is False


def test_account_number_points_accept_known_match(
    monkeypatch: pytest.MonkeyPatch, points_cfg: account_merge.MergeCfg
) -> None:
    def fake_match_account_number_best_pair(*_: Any, **__: Any) -> tuple[bool, Dict[str, Any]]:
        return True, {
            "acctnum_level": "known_match",
            "matched": True,
            "matched_bool": True,
            "match_score": 1.0,
            "best_pair": ("transunion", "experian"),
            "normalized_values": ("2222", "2222"),
            "raw_values": {"a": "2222", "b": "2222"},
        }

    monkeypatch.setattr(
        account_merge,
        "_match_account_number_best_pair",
        fake_match_account_number_best_pair,
    )

    bureaus_a = _make_bureaus(transunion=_all_signals_payload())
    bureaus_b = _make_bureaus(experian=_all_signals_payload())

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    expected_weight = points_cfg.weights["account_number"]
    assert scored["parts"]["account_number"] == pytest.approx(expected_weight)
    account_aux = scored["aux"]["account_number"]
    assert account_aux["points_mode_acctnum_level"] == "known_match"
    assert account_aux["points_mode_matched_bool"] is True
