import importlib
from pathlib import Path

import pytest

import backend.config as config
from backend.core.case_store import api as cs_api, storage
from backend.core.case_store.models import AccountCase, AccountFields, Bureau
from backend.core.logic.report_analysis.extractors import accounts
from backend.core.logic.report_analysis import problem_detection as pd
import backend.core.orchestrators as orch
from backend.core.orchestrators import compute_logical_account_key

from tests.helpers.case_asserts import dict_superset, list_merge_preserves

BUREAUS = {"EX": "Experian", "EQ": "Equifax", "TU": "TransUnion"}


def _setup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "CASESTORE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    monkeypatch.setattr(config, "ENABLE_CASESTORE_STAGEA", True)
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)
    monkeypatch.setattr(config, "ENABLE_CROSS_BUREAU_RESOLUTION", True)
    monkeypatch.setattr(config, "ENABLE_CANDIDATE_TOKEN_LOGGER", False)
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1")
    monkeypatch.setenv("ONE_CASE_PER_ACCOUNT_ENABLED", "1")
    monkeypatch.setenv("NORMALIZED_OVERLAY_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    import backend.core.config.flags as flags
    importlib.reload(flags)
    importlib.reload(cs_api)
    importlib.reload(accounts)
    importlib.reload(pd)
    importlib.reload(orch)


def _create_session(monkeypatch, tmp_path: Path) -> str:
    _setup(monkeypatch, tmp_path)
    session_id = "sess1"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    return session_id


def _lines(balance: int) -> list[str]:
    return [
        "Account # 123456789",
        "Creditor Type: Bank",
        "Date Opened: 2020-01-01",
        f"Balance Owed: ${balance}",
        "Credit Limit: $1000",
    ]


def _extract_all(session_id: str, bureaus: list[str] | None = None) -> None:
    bureaus = bureaus or list(BUREAUS.values())
    mapping = {"Experian": 100, "Equifax": 200, "TransUnion": 300}
    for b in bureaus:
        accounts.extract(_lines(mapping[b]), session_id=session_id, bureau=b)


def _fake_eval_with_optional_ai(session_id, account_id, fields, doc_fp, acct_fp):
    bal = fields.get("balance_owed")
    mapping = {100: ("Tier3", 0.4), 200: ("Tier2", 0.6), 300: ("Tier2", 0.9)}
    tier, conf = mapping.get(bal, ("none", 0.0))
    decision = {
        "primary_issue": "collection",
        "tier": tier,
        "confidence": conf,
        "decision_source": "ai",
        "problem_reasons": [],
    }
    return decision, True, 0.0, None, conf


def _run_stage_a(session_id: str, monkeypatch) -> None:
    monkeypatch.setattr(pd, "evaluate_with_optional_ai", _fake_eval_with_optional_ai)
    pd.run_stage_a(session_id)


def _single_account_id(session_id: str) -> str:
    case = cs_api.load_session_case(session_id)
    assert len(case.accounts) == 1
    return next(iter(case.accounts))


def test_e2e_extraction_consolidates_by_bureau(tmp_path, monkeypatch):
    session_id = _create_session(monkeypatch, tmp_path)
    _extract_all(session_id)
    account_id = _single_account_id(session_id)

    temp_case = AccountCase(
        bureau=Bureau.Equifax,
        fields=AccountFields(
            account_number="123456789",
            creditor_type="Bank",
            date_opened="2020-01-01",
        ),
    )
    logical_key = compute_logical_account_key(temp_case)
    case = cs_api.load_session_case(session_id)
    assert case.summary.logical_index[logical_key] == account_id

    acc_case = cs_api.get_account_case(session_id, account_id)
    by_bureau = acc_case.fields.model_dump().get("by_bureau", {})
    assert set(by_bureau.keys()) == {"EX", "EQ", "TU"}
    for code in ["EX", "EQ", "TU"]:
        fields = by_bureau[code]
        assert fields.get("balance_owed") is not None
        assert fields.get("credit_limit") is not None
        assert fields.get("date_opened") == "2020-01-01"


def test_e2e_stagea_writes_per_bureau_artifacts_and_legacy_winner(tmp_path, monkeypatch):
    session_id = _create_session(monkeypatch, tmp_path)
    _extract_all(session_id)
    _run_stage_a(session_id, monkeypatch)
    account_id = _single_account_id(session_id)
    case = cs_api.get_account_case(session_id, account_id)
    artifacts = case.artifacts
    for code in ["EX", "EQ", "TU"]:
        art = artifacts.get(f"stageA_detection.{code}")
        assert art is not None
        payload = art.model_dump()
        for key in [
            "primary_issue",
            "tier",
            "problem_reasons",
            "decision_source",
            "confidence",
        ]:
            assert key in payload
    assert "stageA_detection" in artifacts
    legacy = artifacts["stageA_detection"].model_dump()
    winner = artifacts["stageA_detection.TU"].model_dump()
    legacy.pop("timestamp", None)
    winner.pop("timestamp", None)
    winner.pop("bureau", None)
    assert legacy == winner


def test_e2e_collectors_emit_stable_shape(tmp_path, monkeypatch):
    session_id = _create_session(monkeypatch, tmp_path)
    _extract_all(session_id)
    _run_stage_a(session_id, monkeypatch)
    rows = orch.collect_stageA_problem_accounts(session_id)
    assert len(rows) == 3
    bureaus = {r["bureau"] for r in rows}
    assert bureaus == {"EX", "EQ", "TU"}
    required = {
        "account_id",
        "bureau",
        "primary_issue",
        "tier",
        "problem_reasons",
        "confidence",
        "decision_source",
    }
    for r in rows:
        assert set(r.keys()) >= required
        assert r["account_id"] == _single_account_id(session_id)
    winners = orch.collect_stageA_logical_accounts(session_id)
    assert len(winners) == 1
    w = winners[0]
    assert w["bureau"] == "TU"
    for key in required - {"bureau", "account_id"}:
        assert key in w
    case = cs_api.get_account_case(session_id, _single_account_id(session_id))
    legacy = case.artifacts["stageA_detection"].model_dump()
    assert w["tier"] == legacy["tier"]
    assert w["confidence"] == legacy["confidence"]
    assert w["decision_source"] == legacy["decision_source"]
    assert w["primary_issue"] == legacy["primary_issue"]


def test_e2e_idempotent_rerun_no_data_loss(tmp_path, monkeypatch):
    session_id = _create_session(monkeypatch, tmp_path)
    _extract_all(session_id)
    account_id = _single_account_id(session_id)
    cs_api.upsert_account_fields(
        session_id,
        account_id,
        "Experian",
        {
            "by_bureau": {
                "EX": {"payment_history": [{"date": "2023-01", "status": "OK"}]}
            }
        },
    )
    _run_stage_a(session_id, monkeypatch)
    before = cs_api.get_account_case(session_id, account_id).fields.model_dump()
    _extract_all(session_id)
    _run_stage_a(session_id, monkeypatch)
    after = cs_api.get_account_case(session_id, account_id).fields.model_dump()
    dict_superset(after, before)
    list_merge_preserves(
        before["by_bureau"]["EX"]["payment_history"],
        after["by_bureau"]["EX"]["payment_history"],
        key="date",
    )


def test_e2e_partial_bureaus_then_additional_bureau(tmp_path, monkeypatch):
    session_id = _create_session(monkeypatch, tmp_path)
    _extract_all(session_id, bureaus=["Experian", "TransUnion"])
    account_id = _single_account_id(session_id)
    before = cs_api.get_account_case(session_id, account_id).fields.model_dump()
    assert set(before["by_bureau"].keys()) == {"EX", "TU"}
    _extract_all(session_id)  # now includes Equifax
    after = cs_api.get_account_case(session_id, account_id).fields.model_dump()
    assert set(after["by_bureau"].keys()) == {"EX", "EQ", "TU"}
    for code in ["EX", "TU"]:
        assert after["by_bureau"][code] == before["by_bureau"][code]


BUREAU_NAMES = {"EX": "Experian", "EQ": "Equifax", "TU": "TransUnion"}


def _setup_legacy(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(config, "CASESTORE_DIR", tmp_path.as_posix())
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1")
    monkeypatch.setenv("ONE_CASE_PER_ACCOUNT_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    import backend.core.config.flags as flags
    importlib.reload(flags)
    importlib.reload(cs_api)
    import backend.core.compat.legacy_shim as shim
    importlib.reload(shim)
    import backend.core.materialize.casestore_view as cs_view
    importlib.reload(cs_view)
    import backend.api.app as app_module
    importlib.reload(app_module)
    return app_module


def _create_legacy_session(session_id: str) -> None:
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    base = {
        "account_number": "00001234",
        "creditor_type": "card",
        "date_opened": "2020-01-01",
    }
    for idx, code in enumerate(["EX", "EQ", "TU"], 1):
        fields = dict(base)
        fields["balance_owed"] = idx
        acc_id = f"acc-{code.lower()}"
        cs_api.upsert_account_fields(session_id, acc_id, BUREAU_NAMES[code], fields)
        cs_api.append_artifact(session_id, acc_id, "stageA_detection", {"tier": "none"})


def test_e2e_legacy_session_shim_visibility(tmp_path, monkeypatch):
    app_module = _setup_legacy(monkeypatch, tmp_path)
    session_id = "sess-legacy"
    _create_legacy_session(session_id)
    app = app_module.create_app()
    client = app.test_client()
    resp = client.get(f"/api/account/{session_id}/acc-ex")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data["fields"]["by_bureau"].keys()) == {"EX", "EQ", "TU"}
    assert "stageA_detection" in data["artifacts"]
    assert not any(k.startswith("stageA_detection.") for k in data["artifacts"] if k != "stageA_detection")
    assert data["meta"]["present_bureaus"] == ["EQ", "EX", "TU"]
    case = cs_api.get_account_case(session_id, "acc-ex")
    assert getattr(case.fields, "by_bureau", None) is None
