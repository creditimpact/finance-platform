from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Callable

import pytest

from backend.ai.merge import sender
from backend.config import merge_config
from backend.config.merge_config import POINTS_MODE_DEFAULT_WEIGHTS
from backend.core.logic.report_analysis import account_merge


def _make_bureaus(**kwargs: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    base = {"transunion": {}, "experian": {}, "equifax": {}}
    for bureau, values in kwargs.items():
        base[bureau] = values
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
        fields=account_merge._POINTS_MODE_FIELD_ALLOWLIST,
        overrides={},
        allowlist_enforce=True,
        allowlist_fields=account_merge._POINTS_MODE_FIELD_ALLOWLIST,
    )
    setattr(cfg, "points_mode", True)
    setattr(cfg, "ai_points_threshold", 3.0)
    setattr(cfg, "direct_points_threshold", 5.0)
    setattr(cfg, "field_sequence", account_merge._POINTS_MODE_FIELD_ALLOWLIST)
    setattr(cfg, "weights_map", dict(weights))
    return cfg


@pytest.fixture()
def points_cfg() -> account_merge.MergeCfg:
    return _build_points_cfg()


@pytest.fixture()
def make_points_accounts() -> Callable[
    ..., tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]
]:
    base_fields = {
        "account_number": "1234567890",
        "date_opened": "2021-03-15",
        "account_type": "installment",
        "account_status": "open",
        "balance_owed": "100",
        "history_2y": "OK OK",
        "history_7y": "OK OK",
    }

    def _factory(
        *,
        left_override: dict[str, object] | None = None,
        right_override: dict[str, object] | None = None,
    ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
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


def test_points_mode_account_number_adds_single_point(points_cfg) -> None:
    bureaus_a = _make_bureaus(transunion={"account_number": "1234"})
    bureaus_b = _make_bureaus(experian={"account_number": "1234"})

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    assert isinstance(scored["total"], float)
    assert scored["total"] == pytest.approx(1.0)
    assert scored["field_contributions"]["account_number"] == pytest.approx(1.0)


def test_points_mode_balance_within_tolerance(points_cfg) -> None:
    custom_cfg = _build_points_cfg()
    custom_cfg.tolerances = dict(custom_cfg.tolerances)
    custom_cfg.tolerances.update({"MERGE_TOL_BALANCE_ABS": 5.0})

    bureaus_a = _make_bureaus(transunion={"balance_owed": "100"})
    bureaus_b = _make_bureaus(experian={"balance_owed": "103"})

    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, custom_cfg)

    assert scored["total"] == pytest.approx(3.0)
    assert scored["field_contributions"]["balance_owed"] == pytest.approx(3.0)


def test_points_mode_ai_and_direct_thresholds(points_cfg, make_points_accounts) -> None:
    bureaus_a, bureaus_b = make_points_accounts(
        left_override={
            "history_2y": "OK OK",
            "history_7y": "OK OK",
            "balance_owed": "100",
        },
        right_override={
            "history_2y": "LATE",
            "history_7y": "LATE",
            "balance_owed": "250",
        },
    )
    ai_result = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)
    assert ai_result["total"] == pytest.approx(3.0)
    assert ai_result["decision"] == "ai"
    assert "points:ai" in ai_result["triggers"]

    bureaus_a, bureaus_b = make_points_accounts(
        left_override={
            "account_type": "installment",
            "account_status": "open",
            "history_2y": "OK OK",
            "history_7y": "LATE",
        },
        right_override={
            "account_type": "revolving",
            "account_status": "closed",
            "history_2y": "LATE",
            "history_7y": "OK OK",
        },
    )
    bureaus_a["transunion"]["balance_owed"] = "100"
    bureaus_b["experian"]["balance_owed"] = "100"

    auto_result = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)
    assert auto_result["total"] == pytest.approx(5.0)
    assert auto_result["decision"] == "auto"
    assert "points:direct" in auto_result["triggers"]


def test_points_mode_allowlist_filters_unknown_fields(monkeypatch, caplog) -> None:
    monkeypatch.setenv("MERGE_POINTS_MODE", "1")
    monkeypatch.setenv("MERGE_ALLOWLIST_ENFORCE", "1")
    fields = list(account_merge._POINTS_MODE_FIELD_ALLOWLIST) + ["rogue_field"]
    monkeypatch.setenv("MERGE_FIELDS_OVERRIDE_JSON", json.dumps(fields))
    weights = dict(POINTS_MODE_DEFAULT_WEIGHTS)
    weights["rogue_field"] = 9.0
    monkeypatch.setenv("MERGE_WEIGHTS_JSON", json.dumps(weights))
    caplog.set_level("WARNING", logger="backend.config.merge_config")

    merge_config.reset_merge_config_cache()
    try:
        cfg = account_merge.get_merge_cfg()
        bureaus_a = _make_bureaus(transunion={"account_number": "1234"})
        bureaus_b = _make_bureaus(experian={"account_number": "1234"})

        scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, cfg)
        assert set(scored["fields_evaluated"]) == set(account_merge._POINTS_MODE_FIELD_ALLOWLIST)
        assert "rogue_field" not in scored["field_contributions"]
        assert any("Ignoring weights for non-configured fields" in record.message for record in caplog.records)
    finally:
        merge_config.reset_merge_config_cache()


def test_points_mode_serialization_preserves_float_parts(points_cfg) -> None:
    parts = {field: 0.0 for field in account_merge._POINTS_MODE_FIELD_ALLOWLIST}
    parts["account_number"] = 1.0

    normalized = account_merge.normalize_parts_for_serialization(parts, points_cfg, points_mode=True)
    assert isinstance(normalized["account_number"], float)
    assert normalized["account_number"] == pytest.approx(1.0)
    assert all(isinstance(value, float) for value in normalized.values())


def test_merge_pair_tag_serializes_points(points_cfg) -> None:
    bureaus_a = _make_bureaus(transunion={"account_number": "1234"})
    bureaus_b = _make_bureaus(experian={"account_number": "1234"})
    scored = account_merge.score_pair_0_100(bureaus_a, bureaus_b, points_cfg)

    tag = account_merge.build_merge_pair_tag(2, scored)
    assert isinstance(tag["total"], float)
    assert tag["total"] == pytest.approx(1.0)
    assert isinstance(tag["parts"]["account_number"], float)


def test_trigger_autosend_enqueues_when_enabled(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, list[str] | None, dict | None]] = []

    class DummyResult:
        id = "task-123"

    def _fake_send_task(name: str, args: list[str] | None = None, kwargs: dict | None = None):
        calls.append((name, args, kwargs))
        return DummyResult()

    dummy_celery = types.SimpleNamespace(send_task=_fake_send_task)
    monkeypatch.setitem(sys.modules, "backend.api.tasks", types.SimpleNamespace(app=dummy_celery))
    monkeypatch.setattr(sender.config, "MERGE_AUTOSEND", True, raising=False)
    monkeypatch.setattr(sender.config, "MERGE_SEND_ON_BUILD", True, raising=False)

    sender.trigger_autosend_after_build("SID-123", runs_root=tmp_path, created=2)

    assert calls
    task_name, args, kwargs = calls[0]
    assert task_name == "backend.ai.merge.tasks.send_merge_packs"
    assert args == ["SID-123"]
    assert kwargs == {
        "runs_root": str(tmp_path.resolve()),
        "reason": "build",
    }


def test_trigger_autosend_noop_when_disabled(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, list[str] | None, dict | None]] = []

    def _fake_send_task(name: str, args: list[str] | None = None, kwargs: dict | None = None):
        calls.append((name, args, kwargs))
        return types.SimpleNamespace(id="task-123")

    dummy_celery = types.SimpleNamespace(send_task=_fake_send_task)
    monkeypatch.setitem(sys.modules, "backend.api.tasks", types.SimpleNamespace(app=dummy_celery))
    monkeypatch.setattr(sender.config, "MERGE_AUTOSEND", False, raising=False)
    monkeypatch.setattr(sender.config, "MERGE_SEND_ON_BUILD", False, raising=False)

    sender.trigger_autosend_after_build("SID-999", runs_root=tmp_path, created=1)

    assert calls == []

