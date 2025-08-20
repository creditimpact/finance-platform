import pytest
from types import SimpleNamespace

from backend.core.logic.rendering.letter_rendering import render_dispute_letter_html
from backend.analytics.analytics_tracker import reset_counters, get_counters


class Ctx:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data


def _base_context():
    client = SimpleNamespace(full_name="Jane Doe", address_line="123 Main St")
    return {"client": client, "today": "Jan 1, 2024"}


def _render(template, ctx):
    reset_counters()
    html = render_dispute_letter_html(Ctx(ctx), template).html
    counters = get_counters()
    return html, counters


def test_debt_validation_substance():
    ctx = _base_context()
    ctx.update(
        {
            "collector_name": "ABC Collections",
            "account_number_masked": "1234",
            "bureau": "experian",
            "legal_safe_summary": "Under FDCPA §1692g I request validation within 30 days.",
            "days_since_first_contact": "5",
        }
    )
    html, counters = _render("debt_validation_letter_template.html", ctx)
    assert "1692g" in html.lower()
    assert "30" in html
    assert counters.get(
        "letter_template_selected.debt_validation_letter_template.html"
    ) == 1


def test_fraud_dispute_substance():
    ctx = _base_context()
    ctx.update(
        {
            "creditor_name": "XYZ Bank",
            "account_number_masked": "9999",
            "bureau": "experian",
            "legal_safe_summary": "Per FCRA §605B and my FTC report, please block or remove this account and respond within 30 days.",
            "is_identity_theft": True,
        }
    )
    html, _ = _render("fraud_dispute_letter_template.html", ctx)
    low = html.lower()
    assert "605b" in low and "block" in low and "30" in low


def test_mov_substance():
    ctx = _base_context()
    ctx.update(
        {
            "creditor_name": "Bank",
            "account_number_masked": "2222",
            "legal_safe_summary": "Please reinvestigate this account and provide the method of verification.",
            "cra_last_result": "verified",
            "days_since_cra_result": "45",
        }
    )
    html, _ = _render("mov_letter_template.html", ctx)
    assert "reinvestigate" in html.lower()
    assert "verified" in html and "45" in html


def test_pay_for_delete_substance():
    ctx = _base_context()
    ctx.update(
        {
            "collector_name": "Collector",
            "account_number_masked": "3333",
            "legal_safe_summary": "I will pay if you delete the account.",
            "offer_terms": "Pay $100 for deletion",
        }
    )
    html, _ = _render("pay_for_delete_letter_template.html", ctx)
    low = html.lower()
    assert "pay" in low and "delete" in low


def test_cease_and_desist_substance():
    ctx = _base_context()
    ctx.update(
        {
            "collector_name": "Debt Co",
            "account_number_masked": "4444",
            "legal_safe_summary": "Cease all communication and stop contacting me.",
        }
    )
    html, _ = _render("cease_and_desist_letter_template.html", ctx)
    low = html.lower()
    assert "cease" in low and "stop" in low


def test_debt_validation_missing_triggers_failure():
    ctx = _base_context()
    ctx.update(
        {
            "collector_name": "ABC Collections",
            "account_number_masked": "1234",
            "bureau": "experian",
            "legal_safe_summary": "Please validate this debt.",
            "days_since_first_contact": "5",
        }
    )
    reset_counters()
    with pytest.raises(ValueError):
        render_dispute_letter_html(Ctx(ctx), "debt_validation_letter_template.html")
    counters = get_counters()
    assert counters.get(
        "validation.failed.debt_validation_letter_template.html.fdcpa_1692g"
    ) == 1


def test_dispute_missing_triggers_failure():
    ctx = {
        "client_name": "Jane Doe",
        "client_address_lines": ["123 Main St"],
        "bureau_name": "Experian",
        "bureau_address": "Address",
        "date": "Jan 1, 2024",
        "account_number_masked": "1234",
        "opening_paragraph": "Please investigate this account and respond within 30 days.",
        "accounts": [
            {
                "name": "Bank",
                "account_number": "1234",
                "status": "Late",
            }
        ],
        "closing_paragraph": "Thank you",
    }
    reset_counters()
    with pytest.raises(ValueError):
        render_dispute_letter_html(Ctx(ctx), "dispute_letter_template.html")
    counters = get_counters()
    assert counters.get(
        "validation.failed.dispute_letter_template.html.fcra_611"
    ) == 1


def test_goodwill_substance():
    ctx = _base_context()
    ctx.update(
        {
            "creditor_name": "Good Bank",
            "account_number_masked": "5555",
            "bureau": "experian",
            "legal_safe_summary": "I respectfully request a goodwill adjustment based on my positive payment history without admitting responsibility.",
            "months_since_last_late": "12",
            "account_history_good": "I have a positive history with your bank.",
        }
    )
    html, _ = _render("goodwill_letter_template.html", ctx)
    low = html.lower()
    assert "goodwill" in low and "positive" in low and "request" in low


def test_goodwill_missing_triggers_failure():
    ctx = _base_context()
    ctx.update(
        {
            "creditor_name": "Good Bank",
            "account_number_masked": "5555",
            "bureau": "experian",
            "legal_safe_summary": "Please remove this account.",
            "months_since_last_late": "12",
            "account_history_good": "",
        }
    )
    reset_counters()
    with pytest.raises(ValueError):
        render_dispute_letter_html(Ctx(ctx), "goodwill_letter_template.html")
    counters = get_counters()
    assert counters.get(
        "validation.failed.goodwill_letter_template.html.non_promissory_tone"
    ) == 1
