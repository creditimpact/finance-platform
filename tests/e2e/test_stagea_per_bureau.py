import importlib
import json
from pathlib import Path

import pytest

import backend.config as config
from backend.core.case_store import api as cs_api
from backend.core.logic.report_analysis import problem_detection as pd
from backend.core.logic.report_analysis.normalize_fields import ensure_all_keys
from backend.core.logic.report_analysis.problem_case_builder import (
    _build_bureaus_payload_from_stagea,
    build_problem_cases,
)


class ExplodingAccounts:
    def __len__(self):
        return 0

    def __iter__(self):
        raise AssertionError("Stage-A attempted to use in-memory accounts")


def _setup_env(monkeypatch, tmp_path, flag_on: bool) -> None:
    monkeypatch.setattr(config, "CASESTORE_DIR", str(tmp_path))
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", False)
    monkeypatch.setattr(config, "ENABLE_CANDIDATE_TOKEN_LOGGER", False)
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1")
    monkeypatch.setenv("ONE_CASE_PER_ACCOUNT_ENABLED", "1" if flag_on else "0")
    import backend.core.config.flags as flags
    importlib.reload(flags)
    importlib.reload(cs_api)
    importlib.reload(pd)


def _create_session(monkeypatch, tmp_path, flag_on: bool) -> str:
    _setup_env(monkeypatch, tmp_path, flag_on)
    session_id = "sess1"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    fields = {
        "by_bureau": {
            "EX": {"balance_owed": 1, "credit_limit": 1000},
            "EQ": {"balance_owed": 2, "credit_limit": 1000},
            "TU": {"balance_owed": 3, "credit_limit": 1000},
        }
    }
    cs_api.upsert_account_fields(session_id, "acc1", "Experian", fields)
    return session_id


def test_stagea_writes_namespaced_artifacts_per_bureau_when_flag_on(tmp_path, monkeypatch):
    session_id = _create_session(monkeypatch, tmp_path, True)
    pd.run_stage_a(session_id)
    case = cs_api.get_account_case(session_id, "acc1")
    artifacts = case.artifacts
    for code in ["EX", "EQ", "TU"]:
        art = artifacts.get(f"stageA_detection.{code}")
        assert art is not None
        payload = art.model_dump()
        assert payload.get("bureau") == code
        for key in ["primary_issue", "tier", "decision_source"]:
            assert key in payload


def test_stagea_keeps_legacy_winner_during_transition(tmp_path, monkeypatch):
    session_id = _create_session(monkeypatch, tmp_path, True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)

    def fake_eval_with_optional_ai(session_id, account_id, fields, doc_fp, acct_fp):
        bal = fields.get("balance_owed")
        mapping = {
            1: ("Tier3", 0.4),
            2: ("Tier2", 0.6),
            3: ("Tier2", 0.9),
        }
        tier, conf = mapping.get(bal, ("none", 0.0))
        decision = {
            "primary_issue": "collection",
            "tier": tier,
            "confidence": conf,
            "decision_source": "ai",
            "problem_reasons": [],
        }
        return decision, True, 0.0, None, conf

    monkeypatch.setattr(pd, "evaluate_with_optional_ai", fake_eval_with_optional_ai)
    pd.run_stage_a(session_id)
    case = cs_api.get_account_case(session_id, "acc1")
    artifacts = case.artifacts
    assert "stageA_detection" in artifacts
    winner = artifacts["stageA_detection"].model_dump()
    tu = artifacts["stageA_detection.TU"].model_dump()
    tu_copy = dict(tu)
    tu_copy.pop("bureau", None)
    winner.pop("timestamp", None)
    tu_copy.pop("timestamp", None)
    assert winner == tu_copy


def test_stagea_flag_off_preserves_legacy_behavior(tmp_path, monkeypatch):
    session_id = _create_session(monkeypatch, tmp_path, False)
    pd.run_stage_a(session_id)
    case = cs_api.get_account_case(session_id, "acc1")
    artifacts = case.artifacts
    assert "stageA_detection" in artifacts
    assert not any(k.startswith("stageA_detection.") for k in artifacts if k != "stageA_detection")


def test_stagea_reads_only_case_store(tmp_path, monkeypatch):
    session_id = _create_session(monkeypatch, tmp_path, True)
    pd.run_stage_a(session_id, ExplodingAccounts())
    case = cs_api.get_account_case(session_id, "acc1")
    assert "stageA_detection.EX" in case.artifacts


def test_stagea_bureaus_payload_includes_original_creditor():
    triad_fields = {
        "transunion": ensure_all_keys({"original_creditor": "PALISADES"}),
        "experian": ensure_all_keys({}),
        "equifax": ensure_all_keys({"original_creditor": "EQ VALUE"}),
    }

    account = {"triad_fields": triad_fields}
    bureaus_payload = _build_bureaus_payload_from_stagea(account)

    assert bureaus_payload["transunion"]["original_creditor"] == "PALISADES"
    assert bureaus_payload["experian"]["original_creditor"] == ""
    assert bureaus_payload["equifax"]["original_creditor"] == "EQ VALUE"


def test_stagea_build_problem_cases_writes_original_creditor(tmp_path):
    sid = "sess1"
    accounts_dir = tmp_path / "traces" / "blocks" / sid / "accounts_table"
    accounts_dir.mkdir(parents=True, exist_ok=True)

    triad_fields = {
        "transunion": ensure_all_keys({"original_creditor": "PALISADES FUNDING CORP"}),
        "experian": ensure_all_keys({"original_creditor": ""}),
        "equifax": ensure_all_keys({"original_creditor": ""}),
    }
    two_year = {"transunion": "YYNN", "experian": "", "equifax": ""}
    seven_year = {
        "transunion": {"late30": 1, "late60": 0, "late90": 0, "late120": 0},
        "experian": {},
        "equifax": {},
    }
    account_data = {
        "account_index": 1,
        "triad_fields": triad_fields,
        "two_year_payment_history": two_year,
        "seven_year_history": seven_year,
        "triad": {"order": ["transunion", "experian", "equifax"]},
        "lines": [],
    }
    accounts_path = accounts_dir / "accounts_from_full.json"
    accounts_path.write_text(
        json.dumps({"accounts": [account_data]}), encoding="utf-8"
    )

    candidates = [{"account_index": 1, "account_id": "acc1"}]
    build_problem_cases(sid, candidates, root=tmp_path)

    bureaus_path = tmp_path / "runs" / sid / "cases" / "accounts" / "1" / "bureaus.json"
    payload = json.loads(bureaus_path.read_text(encoding="utf-8"))

    assert payload["transunion"]["original_creditor"] == "PALISADES FUNDING CORP"
    assert payload["experian"]["original_creditor"] == ""
    assert payload["equifax"]["original_creditor"] == ""
    assert payload["order"] == ["transunion", "experian", "equifax"]
    assert payload["two_year_payment_history"] == two_year
    assert payload["seven_year_history"] == seven_year
