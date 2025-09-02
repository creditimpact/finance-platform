import importlib

import pytest

from backend.core.case_store import api, storage
from backend.core.case_store.models import AccountCase, AccountFields, Bureau
from backend.core.logic.report_analysis.extractors import accounts
from backend.core.orchestrators import compute_logical_account_key
from tests.helpers.case_asserts import dict_superset, list_merge_preserves


def setup_case(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CASESTORE_DIR", tmp_path.as_posix())
    case = api.create_session_case("sess")
    api.save_session_case(case)
    return case.session_id


def _lines(balance: int) -> list[str]:
    return [
        "Account # 123456789",
        "Creditor Type: Bank",
        "Date Opened: 2020-01-01",
        f"Balance Owed: ${balance}",
        "Credit Limit: $1000",
    ]


def test_extraction_populates_by_bureau_three_ways(tmp_path, monkeypatch):
    monkeypatch.setenv("ONE_CASE_PER_ACCOUNT_ENABLED", "1")
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1")
    import backend.core.config.flags as flags
    importlib.reload(flags)
    importlib.reload(api)
    importlib.reload(accounts)

    session_id = setup_case(tmp_path, monkeypatch)

    accounts.extract(_lines(100), session_id=session_id, bureau="Experian")
    accounts.extract(_lines(200), session_id=session_id, bureau="Equifax")
    accounts.extract(_lines(300), session_id=session_id, bureau="TransUnion")

    case = api.load_session_case(session_id)
    assert len(case.accounts) == 1
    account_id = next(iter(case.accounts))

    temp_case = AccountCase(
        bureau=Bureau.Equifax,
        fields=AccountFields(
            account_number="123456789",
            creditor_type="Bank",
            date_opened="2020-01-01",
        ),
    )
    logical_key = compute_logical_account_key(temp_case)
    assert case.summary.logical_index[logical_key] == account_id

    acc_case = api.get_account_case(session_id, account_id)
    by_bureau = acc_case.fields.model_dump().get("by_bureau", {})
    expected_balances = {"EX": 100, "EQ": 200, "TU": 300}
    for code, bal in expected_balances.items():
        assert by_bureau[code]["balance_owed"] == bal
        assert by_bureau[code]["credit_limit"] is not None
        assert by_bureau[code]["date_opened"] == "2020-01-01"


def test_extraction_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("ONE_CASE_PER_ACCOUNT_ENABLED", "1")
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1")
    import backend.core.config.flags as flags
    importlib.reload(flags)
    importlib.reload(api)
    importlib.reload(accounts)

    session_id = setup_case(tmp_path, monkeypatch)

    accounts.extract(_lines(100), session_id=session_id, bureau="Experian")
    accounts.extract(_lines(200), session_id=session_id, bureau="Equifax")
    accounts.extract(_lines(300), session_id=session_id, bureau="TransUnion")

    account_id = next(iter(api.load_session_case(session_id).accounts))

    api.upsert_account_fields(
        session_id=session_id,
        account_id=account_id,
        bureau="Experian",
        fields={
            "by_bureau": {
                "EX": {"payment_history": [{"date": "2023-01", "status": "OK"}]}
            }
        },
    )

    before = api.get_account_case(session_id, account_id).fields.model_dump()

    accounts.extract(_lines(100), session_id=session_id, bureau="Experian")
    accounts.extract(_lines(200), session_id=session_id, bureau="Equifax")
    accounts.extract(_lines(300), session_id=session_id, bureau="TransUnion")

    after = api.get_account_case(session_id, account_id).fields.model_dump()

    dict_superset(after, before)
    list_merge_preserves(
        before["by_bureau"]["EX"]["payment_history"],
        after["by_bureau"]["EX"]["payment_history"],
        key="date",
    )


def test_flag_off_preserves_legacy_behavior(tmp_path, monkeypatch):
    monkeypatch.setenv("ONE_CASE_PER_ACCOUNT_ENABLED", "0")
    monkeypatch.setenv("SAFE_MERGE_ENABLED", "1")
    import backend.core.config.flags as flags
    importlib.reload(flags)
    importlib.reload(api)
    importlib.reload(accounts)

    session_id = setup_case(tmp_path, monkeypatch)

    def lines(num: str) -> list[str]:
        return [f"Account # {num}", "Balance Owed: $100"]

    accounts.extract(lines("123456781"), session_id=session_id, bureau="Experian")
    accounts.extract(lines("123456782"), session_id=session_id, bureau="Equifax")
    accounts.extract(lines("123456783"), session_id=session_id, bureau="TransUnion")

    case = api.load_session_case(session_id)
    assert len(case.accounts) == 3
    for acc in case.accounts.values():
        assert getattr(acc.fields, "by_bureau", None) is None
