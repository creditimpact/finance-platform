from backend.core.letters.field_population import apply_field_fillers
from backend.analytics.analytics_tracker import reset_counters, get_counters


def test_critical_field_failure_emits_metric_and_defers():
    reset_counters()
    ctx = {"action_tag": "pay_for_delete"}
    apply_field_fillers(ctx)
    counters = get_counters()
    assert "name" in ctx["missing_fields"]
    assert ctx.get("defer_action_tag") is True
    assert counters.get("fields.populate_errors.field.name") == 1
    assert counters.get("fields.populate_errors.tag.pay_for_delete")


def test_optional_field_failure_does_not_defer():
    reset_counters()
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
    counters = get_counters()
    assert "days_since_cra_result" in ctx["missing_fields"]
    assert ctx.get("defer_action_tag") is not True
    assert counters.get("fields.populate_errors.field.days_since_cra_result") == 1
    assert counters.get("fields.populate_errors.tag.mov")
