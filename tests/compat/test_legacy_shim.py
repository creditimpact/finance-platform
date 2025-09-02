import importlib
from pathlib import Path

import backend.config as config
from backend.core.case_store import api as cs_api
import backend.core.compat.legacy_shim as shim


BUREAU_NAMES = {"EX": "Experian", "EQ": "Equifax", "TU": "TransUnion"}


def _setup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "CASESTORE_DIR", tmp_path.as_posix())
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1")
    monkeypatch.setenv("ONE_CASE_PER_ACCOUNT_ENABLED", "1")
    import backend.core.config.flags as flags
    importlib.reload(flags)
    importlib.reload(cs_api)
    importlib.reload(shim)


def _create_legacy_cases(session_id: str, bureaus: list[str]) -> None:
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    base = {
        "account_number": "00001234",
        "creditor_type": "card",
        "date_opened": "2020-01-01",
    }
    for idx, code in enumerate(bureaus, 1):
        fields = dict(base)
        fields["balance_owed"] = idx
        cs_api.upsert_account_fields(session_id, f"acc-{code.lower()}", BUREAU_NAMES[code], fields)


def test_build_by_bureau_shim_merges_three_bureaus(tmp_path, monkeypatch):
    _setup(monkeypatch, tmp_path)
    session_id = "s1"
    _create_legacy_cases(session_id, ["EX", "EQ", "TU"])
    result = shim.build_by_bureau_shim(session_id, "acc-ex")
    assert set(result.keys()) == {"EX", "EQ", "TU"}
    assert result["EX"]["balance_owed"] == 1
    assert result["EQ"]["balance_owed"] == 2
    assert result["TU"]["balance_owed"] == 3


def test_build_by_bureau_shim_handles_partial_bureaus(tmp_path, monkeypatch):
    _setup(monkeypatch, tmp_path)
    session_id = "s2"
    _create_legacy_cases(session_id, ["EX", "TU"])
    result = shim.build_by_bureau_shim(session_id, "acc-ex")
    assert set(result.keys()) == {"EX", "TU"}
    assert "EQ" not in result


def test_build_by_bureau_shim_no_key_returns_empty(tmp_path, monkeypatch):
    _setup(monkeypatch, tmp_path)
    session_id = "s3"
    case = cs_api.create_session_case(session_id)
    cs_api.save_session_case(case)
    cs_api.upsert_account_fields(session_id, "acc-ex", "Experian", {"balance_owed": 10})
    result = shim.build_by_bureau_shim(session_id, "acc-ex")
    assert result == {}
