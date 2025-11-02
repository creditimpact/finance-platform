import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from backend.core.logic.report_analysis import account_merge
from backend.config.merge_config import POINTS_MODE_DEFAULT_WEIGHTS


_ALLOWED_FIELDS = (
    "account_number",
    "date_opened",
    "balance_owed",
    "account_type",
    "account_status",
    "history_2y",
    "history_7y",
)


def _make_bureaus(**kwargs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    base: Dict[str, Dict[str, Any]] = {
        "transunion": {},
        "experian": {},
        "equifax": {},
    }
    for bureau, values in kwargs.items():
        base[bureau] = dict(values)
    return base


def _all_signals_payload(**overrides: Any) -> Dict[str, Any]:
    payload = {
        "account_number": "1234567890",
        "date_opened": "2021-02-02",
        "balance_owed": "1000",
        "account_type": "installment",
        "account_status": "open",
        "history_2y": "OK OK",
        "history_7y": "OK OK",
    }
    payload.update(overrides)
    return payload


def _build_points_cfg() -> account_merge.MergeCfg:
    weights = {field: float(POINTS_MODE_DEFAULT_WEIGHTS[field]) for field in _ALLOWED_FIELDS}
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
        fields=_ALLOWED_FIELDS,
        overrides={},
        allowlist_enforce=True,
        allowlist_fields=_ALLOWED_FIELDS,
    )
    setattr(cfg, "points_mode", True)
    setattr(cfg, "field_sequence", _ALLOWED_FIELDS)
    setattr(cfg, "weights_map", dict(weights))
    setattr(cfg, "ai_points_threshold", 3.0)
    setattr(cfg, "direct_points_threshold", 5.0)
    return cfg


@pytest.mark.parametrize(
    "part_value, aux_entry, expected_reason",
    [
        (1.0, {"matched": False}, "positive_part_without_match"),
        (-1.0, {"matched": False}, "part_without_match"),
        (0.0, "invalid", "aux_not_mapping"),
        (0.0, {"matched": True}, "aux_mismatch"),
    ],
)
def test_points_mode_invariant_logging_emits_context(
    part_value: float,
    aux_entry: Any,
    expected_reason: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    parts_map = {"balance_owed": part_value}
    matched_map = {"balance_owed": False}

    caplog.set_level(logging.ERROR)
    with pytest.raises(AssertionError):
        account_merge._ensure_points_mode_field_invariants(
            field="balance_owed",
            matched_flag=False,
            part_value=part_value,
            parts_map=parts_map,
            matched_map=matched_map,
            aux_entry=aux_entry,
        )

    failure_logs = [
        record
        for record in caplog.records
        if "Points-mode invariant failure" in record.getMessage()
    ]
    assert failure_logs, "Expected invariant failure log to be emitted"
    last_record = failure_logs[-1]
    context = last_record.args[0] if isinstance(last_record.args, tuple) else last_record.args
    assert context["field"] == "balance_owed"
    assert context["reason"] == expected_reason
    assert context["matched"] is False
    assert context["parts"]["balance_owed"] == pytest.approx(parts_map["balance_owed"])


def test_points_mode_pair_logging_snapshot_highlights_pairs() -> None:
    aux_payload = {
        "by_field_pairs": {
            "balance_owed": ("transunion", "experian"),
            "date_opened": ("equifax", "experian"),
        },
        "matched_fields": {field: True for field in _ALLOWED_FIELDS},
    }
    parts = {field: float(POINTS_MODE_DEFAULT_WEIGHTS[field]) for field in _ALLOWED_FIELDS}
    result = {
        "decision": "ai",
        "conflicts": ["amount_conflict:balance_owed"],
        "triggers": ["points:ai"],
    }

    snapshot = account_merge._build_pair_logging_snapshot(
        sid="SID",
        left_index=1,
        right_index=2,
        total=6.5,
        result=result,
        parts=parts,
        aux_payload=aux_payload,
        acctnum_level="exact_or_known_match",
        points_mode=True,
    )

    assert snapshot["sid"] == "SID"
    assert snapshot["acctnum_level"] == "exact_or_known_match"
    assert snapshot["parts"]["balance_owed"] == pytest.approx(parts["balance_owed"])
    assert snapshot["aux_pairs"]["balance_owed"] == ["transunion", "experian"]
    assert snapshot["aux_pairs"]["date_opened"] == ["equifax", "experian"]
    assert snapshot["matched_fields"]["balance_owed"] is True
    assert snapshot["decision"] == "ai"
    assert snapshot["triggers"] == ["points:ai"]


def test_points_mode_runtime_snapshot_logs_allowlist(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _build_points_cfg()
    cfg.debug = True
    cfg.ai_points_threshold = 100.0
    cfg.direct_points_threshold = 100.0

    bureaus_left = _make_bureaus(transunion=_all_signals_payload(balance_owed="2000"))
    bureaus_right = _make_bureaus(experian=_all_signals_payload(balance_owed="2000"))

    sample_accounts = {0: bureaus_left, 1: bureaus_right}

    monkeypatch.setattr(account_merge, "get_merge_cfg", lambda: cfg)
    monkeypatch.setattr(
        account_merge,
        "get_merge_config",
        lambda: {"_present_keys": set(), "enabled": True},
    )
    monkeypatch.setattr(account_merge, "load_bureaus", lambda sid, idx, runs_root=None: sample_accounts[idx])
    monkeypatch.setattr(account_merge, "_configure_candidate_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(account_merge, "start_span", lambda *args, **kwargs: None)
    monkeypatch.setattr(account_merge, "span_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(account_merge, "end_span", lambda *args, **kwargs: None)
    monkeypatch.setattr(account_merge, "runflow_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(account_merge, "steps_pair_topn", lambda *args, **kwargs: 0)
    monkeypatch.setattr(account_merge, "build_ai_pack_for_pair", lambda *args, **kwargs: None)

    packs_dir = tmp_path / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    log_file = tmp_path / "logs.txt"
    index_file = tmp_path / "index.json"

    monkeypatch.setattr(
        account_merge,
        "get_merge_paths",
        lambda runs_root, sid, create=True: SimpleNamespace(
            packs_dir=packs_dir,
            log_file=log_file,
            index_file=index_file,
        ),
    )

    caplog.set_level(logging.DEBUG)

    accounts_root = tmp_path / "SID" / "cases" / "accounts"
    for idx in sample_accounts:
        (accounts_root / str(idx)).mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(
        sys.modules,
        "backend.ai.merge.sender",
        SimpleNamespace(trigger_autosend_after_build=lambda *args, **kwargs: None),
    )

    account_merge.score_all_pairs_0_100("SID", [0, 1], runs_root=tmp_path)

    snapshot_records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("[MERGE] Runtime snapshot")
    ]
    assert snapshot_records, "Expected runtime snapshot log"
    snapshot_record = snapshot_records[-1]
    snapshot_payload = (
        snapshot_record.args[0]
        if isinstance(snapshot_record.args, tuple)
        else snapshot_record.args
    )
    assert snapshot_payload["points_mode"] is True
    assert snapshot_payload["allowlist_fields"] == list(_ALLOWED_FIELDS)
    assert isinstance(snapshot_payload["MERGE_PACKS_DIR"], str)
    assert isinstance(snapshot_payload["MERGE_INDEX_PATH"], str)
    assert isinstance(snapshot_payload["MERGE_PACK_GLOB"], str)
    assert snapshot_payload["ai_points_threshold"] == pytest.approx(cfg.ai_points_threshold)
    assert snapshot_payload["direct_points_threshold"] == pytest.approx(cfg.direct_points_threshold)

    score_records = [
        record
        for record in caplog.records
        if record.getMessage().startswith("MERGE_V2_SCORE ")
    ]
    assert score_records, "Expected MERGE_V2_SCORE log entry"
    score_payload = json.loads(score_records[-1].getMessage().split(" ", 1)[1])
    assert score_payload["acctnum_level"] == "exact_or_known_match"
    assert set(score_payload["parts"].keys()) == set(_ALLOWED_FIELDS)
    assert score_payload["aux_pairs"]["balance_owed"] == ["transunion", "experian"]
    assert score_payload["aux_pairs"]["date_opened"] == ["transunion", "experian"]
    assert score_payload["matched_fields"]["balance_owed"] is True
