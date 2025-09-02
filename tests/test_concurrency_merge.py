from copy import deepcopy

import pytest

from backend.core.case_store import api
from backend.core.case_store.models import AccountCase, AccountFields, Bureau
from backend.core.case_store.errors import CaseWriteConflict
from backend.core.config.flags import Flags


class FakeStore:
    def __init__(self):
        self.db = {}
        self.overrides = []

    def read(self, session_id):
        if self.overrides:
            return deepcopy(self.overrides.pop(0))
        return deepcopy(self.db[session_id])

    def write(self, session_id, case):
        self.db[session_id] = deepcopy(case)


def setup_store(monkeypatch):
    store = FakeStore()
    monkeypatch.setattr(api, "_load", lambda sid: store.read(sid))
    monkeypatch.setattr(api, "_save", lambda case: store.write(case.session_id, case))
    return store


def enable_safe_merge(monkeypatch):
    monkeypatch.setattr(
        api,
        "FLAGS",
        Flags(safe_merge_enabled=True, normalized_overlay_enabled=False, case_first_build_enabled=False),
    )


def test_no_conflict_single_writer(monkeypatch):
    store = setup_store(monkeypatch)
    enable_safe_merge(monkeypatch)

    session_id = "s1"
    account_id = "a1"

    case = api.create_session_case(session_id)
    case.accounts[account_id] = AccountCase(bureau=Bureau.Equifax)
    store.write(session_id, case)

    api.upsert_account_fields(session_id, account_id, "Equifax", {"balance_owed": 100})

    result = store.read(session_id).accounts[account_id]
    assert result.version == 1
    assert result.fields.balance_owed == 100


def test_concurrent_updates_merge_preserved(monkeypatch):
    store = setup_store(monkeypatch)
    enable_safe_merge(monkeypatch)

    session_id = "s2"
    account_id = "a2"
    case = api.create_session_case(session_id)
    case.accounts[account_id] = AccountCase(bureau=Bureau.Equifax)
    store.write(session_id, case)

    stale_case = store.read(session_id)

    api.upsert_account_fields(
        session_id, account_id, "Equifax", {"past_due_amount": 50}
    )

    store.overrides = [stale_case]
    api.upsert_account_fields(
        session_id, account_id, "Equifax", {"credit_limit": 1000}
    )

    final = store.read(session_id).accounts[account_id]
    assert final.version == 2
    assert final.fields.past_due_amount == 50
    assert final.fields.credit_limit == 1000


def test_list_merge_with_conflict(monkeypatch):
    store = setup_store(monkeypatch)
    enable_safe_merge(monkeypatch)

    session_id = "s3"
    account_id = "a3"
    base_case = api.create_session_case(session_id)
    base_case.accounts[account_id] = AccountCase(
        bureau=Bureau.Equifax,
        fields=AccountFields(
            payment_history=[{"date": "2024-01", "status": "OK"}]
        ),
    )
    store.write(session_id, base_case)

    stale_case = store.read(session_id)

    api.upsert_account_fields(
        session_id,
        account_id,
        "Equifax",
        {"payment_history": [{"date": "2024-01", "status": "OK*"}]},
    )

    store.overrides = [stale_case]
    api.upsert_account_fields(
        session_id,
        account_id,
        "Equifax",
        {"payment_history": [{"date": "2024-02", "status": "LATE"}]},
    )

    final = store.read(session_id).accounts[account_id]
    hist = final.fields.payment_history
    assert final.version == 2
    assert len(hist) == 2
    jan = next(item for item in hist if item["date"] == "2024-01")
    feb = next(item for item in hist if item["date"] == "2024-02")
    assert jan["status"] == "OK*"
    assert feb["status"] == "LATE"


def test_retry_exhaustion_raises(monkeypatch):
    store = setup_store(monkeypatch)
    enable_safe_merge(monkeypatch)

    session_id = "s4"
    account_id = "a4"

    # Prepare overrides with steadily increasing versions to force retry exhaustion
    overrides = []
    for v in range(6):
        c = api.create_session_case(session_id)
        c.accounts[account_id] = AccountCase(bureau=Bureau.Equifax, version=v)
        overrides.append(c)
    store.overrides = overrides

    with pytest.raises(CaseWriteConflict) as exc:
        api.upsert_account_fields(
            session_id, account_id, "Equifax", {"credit_limit": 100}
        )

    assert exc.value.last_seen_version == 5


def test_append_artifact_concurrent(monkeypatch):
    store = setup_store(monkeypatch)
    enable_safe_merge(monkeypatch)

    session_id = "s5"
    account_id = "a5"
    case = api.create_session_case(session_id)
    case.accounts[account_id] = AccountCase(bureau=Bureau.Equifax)
    store.write(session_id, case)

    stale_case = store.read(session_id)

    api.append_artifact(
        session_id, account_id, "stageA_detection", {"primary_issue": "B"}
    )

    store.overrides = [stale_case]
    api.append_artifact(
        session_id, account_id, "stageA_detection", {"primary_issue": "A"}
    )

    final = store.read(session_id).accounts[account_id]
    assert final.version == 2
    assert final.artifacts["stageA_detection"].primary_issue == "A"
    assert final.fields.model_dump(exclude_none=True) == {}

