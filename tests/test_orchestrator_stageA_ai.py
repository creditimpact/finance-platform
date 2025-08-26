from backend.core import orchestrators as orch


def test_emit_called_for_problem_accounts(monkeypatch):
    events = []

    def fake_emit(event, payload):
        events.append((event, payload))

    monkeypatch.setattr(orch, "emit", fake_emit)

    accounts = [
        {
            "normalized_name": "palisades fu",
            "account_number_last4": "1234",
            "decision_source": "ai",
            "primary_issue": "collection",
            "confidence": 0.83,
            "tier": 1,
            "problem_reasons": ["remarks"],
        }
    ]

    orch._emit_stageA_events("sess1", accounts)
    assert events == [
        (
            "stageA_problem_decision",
            {
                "session_id": "sess1",
                "normalized_name": "palisades fu",
                "account_id": "1234",
                "decision_source": "ai",
                "primary_issue": "collection",
                "confidence": 0.83,
                "tier": 1,
                "reasons_count": 1,
            },
        )
    ]
