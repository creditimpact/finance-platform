from backend.core.letters.field_population import apply_field_fillers


def test_critical_field_failure_emits_event_and_defers(monkeypatch):
    events = []
    monkeypatch.setattr(
        "backend.core.letters.field_population.emit_event",
        lambda e, p: events.append((e, p)),
    )
    ctx = {"action_tag": "pay_for_delete"}
    apply_field_fillers(ctx)
    assert "name" in ctx["missing_fields"]
    assert ctx.get("defer_action_tag") is True
    assert any(
        e == "fields.populate_errors" and p["field"] == "name" for e, p in events
    )


def test_optional_field_failure_does_not_defer(monkeypatch):
    events = []
    monkeypatch.setattr(
        "backend.core.letters.field_population.emit_event",
        lambda e, p: events.append((e, p)),
    )
    ctx = {
        "action_tag": "mov",
        "name": "X",
        "address": "123",
        "date_of_birth": "2000-01-01",
        "ssn_masked": "***-**-1234",
        "creditor_name": "Cred",
        "account_number_masked": "****1111",
        "inquiry_creditor_name": "Inq",
        "inquiry_date": "2024-01-01",
    }
    apply_field_fillers(ctx)
    assert "days_since_cra_result" in ctx["missing_fields"]
    assert ctx.get("defer_action_tag") is not True
    assert any(
        e == "fields.populate_errors" and p["field"] == "days_since_cra_result"
        for e, p in events
    )
