import time

import pytest

from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.core.logic.report_analysis.tri_merge_models import Tradeline, TradelineFamily
from services.outcome_ingestion import ingest_report as ingest_module


def _make_family(fid: str, account_id: str, bureau: str, val: object) -> TradelineFamily:
    tl = Tradeline(
        creditor="cred",
        bureau=bureau,
        account_number="123",
        data={"account_id": account_id, "val": val},
    )
    fam = TradelineFamily(account_number="123", tradelines={bureau: tl})
    setattr(fam, "family_id", fid)
    return fam


def test_ingest_report_emits_metrics(monkeypatch):
    reset_counters()
    ingest_module._FAMILY_CACHE.clear()
    monkeypatch.setenv("SESSION_ID", "sess1")

    prev_snap = {
        "fid1": {"experian": {"account_id": "1", "val": 1}},
        "fid2": {"experian": {"account_id": "1", "val": "old"}},
        "fid3": {"experian": {"account_id": "1", "val": "old"}},
    }
    stored = {"tri_merge": {"snapshots": {"1": prev_snap}}}
    monkeypatch.setattr(ingest_module.session_manager, "get_session", lambda sid: stored)
    monkeypatch.setattr(ingest_module.session_manager, "update_session", lambda sid, **kw: None)

    families = [
        _make_family("fid1", "1", "experian", 1),
        _make_family("fid2", "1", "experian", "new"),
        _make_family("fid4", "1", "experian", 4),
    ]
    monkeypatch.setattr(ingest_module, "normalize_and_match", lambda tls: families)
    monkeypatch.setattr(ingest_module, "_extract_tradelines", lambda rpt: [])

    def fake_ingest(sess, event):
        time.sleep(0.001)

    monkeypatch.setattr(ingest_module, "ingest", fake_ingest)

    ingest_module.ingest_report(None, {})

    counters = get_counters()
    assert counters["outcome.verified.bureau.experian"] == 1
    assert counters["outcome.updated.bureau.experian"] == 1
    assert counters["outcome.deleted.bureau.experian"] == 1
    assert counters["outcome.nochange.bureau.experian"] == 1
    assert counters["outcome.ingest_latency_ms"] > 0
    assert counters["outcome.ingest_latency_ms.bureau.experian"] == 4


def test_ingest_errors_metric(monkeypatch):
    reset_counters()
    ingest_module._FAMILY_CACHE.clear()
    monkeypatch.setenv("SESSION_ID", "sess1")

    prev_snap = {"fid1": {"experian": {"account_id": "1", "val": 1}}}
    stored = {"tri_merge": {"snapshots": {"1": prev_snap}}}
    monkeypatch.setattr(ingest_module.session_manager, "get_session", lambda sid: stored)
    monkeypatch.setattr(ingest_module.session_manager, "update_session", lambda sid, **kw: None)

    families = [_make_family("fid1", "1", "experian", 1)]
    monkeypatch.setattr(ingest_module, "normalize_and_match", lambda tls: families)
    monkeypatch.setattr(ingest_module, "_extract_tradelines", lambda rpt: [])

    def failing_ingest(sess, event):  # pragma: no cover - error path
        raise ValueError("boom")

    monkeypatch.setattr(ingest_module, "ingest", failing_ingest)

    with pytest.raises(ValueError):
        ingest_module.ingest_report(None, {})

    counters = get_counters()
    assert counters["outcome.ingest_errors.bureau.experian"] == 1
