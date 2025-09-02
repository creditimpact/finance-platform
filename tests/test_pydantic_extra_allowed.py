def _dump(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def test_accountfields_preserves_unknown(monkeypatch):
    from backend.core.case_store.models import AccountFields

    payload = {
        "balance_owed": 100.0,
        "credit_limit": 1000.0,
        "EXTRA_label_new": "seen-in-report",
        "weird_field": {"nested": 1, "arr": [1, 2]},
    }
    m = AccountFields(**payload)
    out = _dump(m)
    assert out["EXTRA_label_new"] == "seen-in-report"
    assert out["weird_field"]["nested"] == 1
    assert out["weird_field"]["arr"] == [1, 2]

    m2 = AccountFields(**out)
    out2 = _dump(m2)
    assert "EXTRA_label_new" in out2 and "weird_field" in out2


def test_artifact_preserves_unknown():
    from backend.core.case_store.models import Artifact

    payload = {
        "primary_issue": "unknown",
        "tier": "none",
        "decision_source": "rules",
        "ai_meta": {"latency_ms": 12, "trace_id": "abc"},
        "EXTRA_debug_blob": {"foo": "bar"},
    }
    a = Artifact(**payload)
    out = _dump(a)
    assert out["ai_meta"]["latency_ms"] == 12
    assert out["EXTRA_debug_blob"]["foo"] == "bar"

    a2 = Artifact(**out)
    out2 = _dump(a2)
    assert "EXTRA_debug_blob" in out2
