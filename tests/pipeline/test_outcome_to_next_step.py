from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import planner
import services.outcome_ingestion.ingest_report as ingest_mod
from backend.api import session_manager
from backend.core.logic.report_analysis.tri_merge_models import (
    Tradeline,
    TradelineFamily,
)
from backend.core.models import AccountStatus
from backend.outcomes import save_outcome_event
from backend.outcomes.models import Outcome
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


SCENARIOS = [
    (
        [
            _family("f1", 100),
        ],
        Outcome.VERIFIED,
        ["mov", "direct_dispute"],
        AccountStatus.CRA_RESPONDED_VERIFIED,
    ),
    (
        [
            _family("f1", 150),
        ],
        Outcome.UPDATED,
        ["bureau_dispute"],
        AccountStatus.CRA_RESPONDED_UPDATED,
    ),
    (
        [
            _family("f2", 200),
        ],
        Outcome.DELETED,
        [],
        AccountStatus.COMPLETED,
    ),
    (
        [
            _family("f1", 100),
            _family("f2", 200),
        ],
        Outcome.NOCHANGE,
        [],
        AccountStatus.CRA_RESPONDED_NOCHANGE,
    ),
]


def _setup_sessions(monkeypatch):
    store: Dict[str, Dict] = {}

    def fake_get_session(sid):
        return store.get(sid)

    def fake_update_session(sid, **kwargs):
        session = store.setdefault(sid, {})
        session.update(kwargs)
        return session

    monkeypatch.setattr(session_manager, "get_session", fake_get_session)
    monkeypatch.setattr(session_manager, "update_session", fake_update_session)
    monkeypatch.setattr(planner, "get_session", fake_get_session)
    monkeypatch.setattr(planner, "update_session", fake_update_session)
    return store


def test_outcome_to_next_step(monkeypatch):
    store = _setup_sessions(monkeypatch)
    monkeypatch.setenv("SESSION_ID", "s1")

    # avoid auto planner updates during report ingest
    def fake_ingest(sess, event):
        save_outcome_event(sess["session_id"], event)

    monkeypatch.setattr(ingest_mod, "ingest", fake_ingest)

    baseline = [_family("f1", 100)]

    session = {
        "session_id": "s1",
        "strategy": {"accounts": [{"account_id": "1", "action_tag": "dispute"}]},
    }

    for new_fams, outcome, expected_tags, final_status in SCENARIOS:
        calls = {"n": 0}

        def fake_normalize(tls: List[Tradeline]):
            calls["n"] += 1
            return baseline if calls["n"] == 1 else new_fams

        monkeypatch.setattr(ingest_mod, "normalize_and_match", fake_normalize)

        ingest_report(None, {})
        events = ingest_report(None, {})
        event = next(e for e in events if e.outcome == outcome)

        planner.plan_next_step(session, ["dispute"], now=datetime(2024, 1, 1))
        planner.record_send(session, ["1"], now=datetime(2024, 1, 1))

        allowed = planner.handle_outcome(session, event, now=datetime(2024, 1, 10))
        state = planner.load_state(store["s1"]["account_states"]["1"])
        assert state.status == final_status
        assert allowed == expected_tags
        if outcome is Outcome.NOCHANGE:
            assert state.next_eligible_at is not None
        else:
            assert state.next_eligible_at is None

        # reset for next scenario
        store["s1"]["account_states"].pop("1")
        store["s1"]["tri_merge"] = {"snapshots": {}}
