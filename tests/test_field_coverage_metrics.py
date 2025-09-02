import ast
import logging

from backend.core.case_store.models import AccountCase, AccountFields, Bureau
from backend.core.metrics import field_coverage


class FakeMetrics:
    def __init__(self):
        self.calls = []

    def gauge(self, name, value, tags=None):
        self.calls.append(("gauge", name, value, tags))

    def count(self, name, value, tags=None):
        self.calls.append(("count", name, value, tags))


def test_account_coverage_basic(monkeypatch, caplog):
    fake = FakeMetrics()
    monkeypatch.setattr(field_coverage, "metrics", fake)

    fields = {
        "balance_owed": 100,
        "credit_limit": 1000,
        "high_balance": 500,
        "date_opened": "2020-01-01",
        "account_status": "open",
    }

    caplog.set_level(logging.INFO)
    field_coverage.emit_account_field_coverage(
        session_id="S1",
        account_id="A1",
        bureau="Experian",
        fields=fields,
    )

    assert (
        "gauge",
        "stage1.field_coverage.account",
        62,
        {"session_id": "S1", "account_id": "A1", "bureau": "Experian"},
    ) in fake.calls
    missing_record = next(
        r for r in caplog.records if "field_coverage.missing" in r.message
    )
    assert "past_due_amount" in missing_record.message
    assert "payment_status" in missing_record.message
    assert "two_year_payment_history" in missing_record.message


def test_session_summary_top_missing(monkeypatch, caplog):
    fake = FakeMetrics()
    monkeypatch.setattr(field_coverage, "metrics", fake)

    cases = {
        "a1": AccountCase(
            bureau=Bureau.Experian,
            fields=AccountFields(
                balance_owed=1,
                credit_limit=2,
                high_balance=3,
            ),
        ),
        "a2": AccountCase(
            bureau=Bureau.Equifax,
            fields=AccountFields(
                balance_owed=1,
                credit_limit=2,
            ),
        ),
        "a3": AccountCase(
            bureau=Bureau.TransUnion,
            fields=AccountFields(
                balance_owed=1,
                credit_limit=2,
                high_balance=3,
                account_status="open",
            ),
        ),
    }

    monkeypatch.setattr(
        field_coverage.case_store,
        "list_accounts",
        lambda session_id: list(cases.keys()),
    )

    def fake_get_account_case(session_id, account_id):
        return cases[account_id]

    monkeypatch.setattr(
        field_coverage.case_store, "get_account_case", fake_get_account_case
    )

    caplog.set_level(logging.INFO)
    field_coverage.emit_session_field_coverage_summary(session_id="S1")

    counts = {call[3]["field"]: call[2] for call in fake.calls if call[0] == "count"}
    assert counts["date_opened"] == 3
    assert counts["past_due_amount"] == 3

    summary_record = next(
        r for r in caplog.records if "field_coverage.session_summary" in r.message
    )
    payload = ast.literal_eval(summary_record.message.split(" ", 1)[1])
    top_counts = [item["count"] for item in payload["top_missing"]]
    assert top_counts == sorted(top_counts, reverse=True)


def test_no_raise_on_errors(monkeypatch, caplog):
    fake = FakeMetrics()
    monkeypatch.setattr(field_coverage, "metrics", fake)

    monkeypatch.setattr(
        field_coverage.case_store, "list_accounts", lambda session_id: ["a1"]
    )

    def boom(session_id, account_id):
        raise RuntimeError("boom")

    monkeypatch.setattr(field_coverage.case_store, "get_account_case", boom)

    caplog.set_level(logging.ERROR)
    field_coverage.emit_session_field_coverage_summary(session_id="S1")
    assert any("field_coverage_session_failed" in r.message for r in caplog.records)
