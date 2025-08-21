from __future__ import annotations

from typing import Dict, List

import services.outcome_ingestion.ingest_report as ingest_mod
from backend.api import session_manager
from backend.core.logic.report_analysis.tri_merge_models import (
    Tradeline,
    TradelineFamily,
)
from backend.outcomes import load_outcome_history, save_outcome_event
from backend.outcomes.models import Outcome, OutcomeEvent
from services.outcome_ingestion.ingest_report import ingest_report


def _family(fid: str, balance: int) -> TradelineFamily:
    tl = Tradeline(
        creditor="Cred",
        bureau="Experian",
        account_number="1",
        data={"account_id": "1", "balance": balance},
    )
    fam = TradelineFamily(account_number="1", tradelines={"Experian": tl})
    fam.family_id = fid  # type: ignore[attr-defined]
    return fam


def test_ingest_report_emits_all_outcomes(monkeypatch):
    store: Dict[str, Dict] = {}

    def fake_get_session(sid):
        return store.get(sid)

    def fake_update_session(sid, **kwargs):
        session = store.setdefault(sid, {})
        session.update(kwargs)
        return session

    monkeypatch.setattr(session_manager, "get_session", fake_get_session)
    monkeypatch.setattr(session_manager, "update_session", fake_update_session)

    # avoid planner side effects
    def fake_ingest(sess, event: OutcomeEvent) -> None:
        save_outcome_event(sess["session_id"], event)

    monkeypatch.setattr(ingest_mod, "ingest", fake_ingest)
    monkeypatch.setenv("SESSION_ID", "s1")

    baseline = [_family("f1", 100), _family("f2", 200), _family("f3", 300)]
    updated = [_family("f1", 100), _family("f2", 250), _family("f4", 400)]
    calls = {"n": 0}

    def fake_normalize(tls: List[Tradeline]):
        calls["n"] += 1
        return baseline if calls["n"] == 1 else updated

    monkeypatch.setattr(ingest_mod, "normalize_and_match", fake_normalize)

    ingest_report(None, {})
    events = ingest_report(None, {})

    assert {e.outcome for e in events} == {
        Outcome.VERIFIED,
        Outcome.UPDATED,
        Outcome.DELETED,
        Outcome.NOCHANGE,
    }
    by_outcome = {e.outcome: e for e in events}
    assert by_outcome[Outcome.VERIFIED].diff_snapshot is None
    assert by_outcome[Outcome.NOCHANGE].diff_snapshot is None
    assert by_outcome[Outcome.UPDATED].diff_snapshot == {
        "previous": {"Experian": {"account_id": "1", "balance": 200}},
        "current": {"Experian": {"account_id": "1", "balance": 250}},
    }
    assert by_outcome[Outcome.DELETED].diff_snapshot == {
        "previous": {"Experian": {"account_id": "1", "balance": 300}},
        "current": None,
    }

    history = load_outcome_history("s1", "1")
    assert len(history) == 4

    tri_merge = store["s1"]["tri_merge"]["snapshots"]["1"]
    assert set(tri_merge.keys()) == {"f1", "f2", "f4"}
    assert tri_merge["f2"]["Experian"]["balance"] == 250
