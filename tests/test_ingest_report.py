import os

from backend.api import session_manager
from backend.outcomes import load_outcome_history
from services.outcome_ingestion.ingest_report import ingest_report
from backend.outcomes.models import Outcome


def _make_report(accounts):
    return {"Experian": {"negative_accounts": accounts}}


def test_ingest_report_classifies_and_persists(monkeypatch, tmp_path):
    sess_file = tmp_path / "sessions.json"
    monkeypatch.setattr(session_manager, "SESSION_FILE", sess_file)
    monkeypatch.setenv("SESSION_ID", "sess1")

    baseline_accounts = [
        {
            "account_id": "acct1",
            "name": "CredA",
            "account_number": "1111",
            "balance": 100,
            "date_opened": "2020",
            "date_reported": "2024",
        },
        {
            "account_id": "acct1",
            "name": "CredB",
            "account_number": "2222",
            "balance": 200,
            "date_opened": "2020",
            "date_reported": "2024",
        },
        {
            "account_id": "acct1",
            "name": "CredC",
            "account_number": "3333",
            "balance": 300,
            "date_opened": "2020",
            "date_reported": "2024",
        },
    ]
    ingest_report(None, _make_report(baseline_accounts))

    new_accounts = [
        {
            "account_id": "acct1",
            "name": "CredA",
            "account_number": "1111",
            "balance": 100,
            "date_opened": "2020",
            "date_reported": "2024",
        },
        {
            "account_id": "acct1",
            "name": "CredB",
            "account_number": "2222",
            "balance": 250,
            "date_opened": "2020",
            "date_reported": "2024",
        },
        {
            "account_id": "acct1",
            "name": "CredD",
            "account_number": "4444",
            "balance": 400,
            "date_opened": "2020",
            "date_reported": "2024",
        },
    ]
    ingest_report(None, _make_report(new_accounts))

    history = load_outcome_history("sess1", "acct1")
    outcomes = {e.outcome for e in history}
    assert outcomes == {Outcome.VERIFIED, Outcome.UPDATED, Outcome.DELETED, Outcome.NOCHANGE}

    session = session_manager.get_session("sess1")
    assert "tri_merge" in session and "snapshots" in session["tri_merge"]
