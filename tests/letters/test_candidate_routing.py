import pytest

from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.core.letters.router import select_template


PII_FIELDS = {
    "client_name",
    "client_address_lines",
    "date_of_birth",
    "ssn_last4",
    "legal_safe_summary",
    "update_request",
}

INQUIRY_FIELDS = {
    "inquiry_creditor_name",
    "account_number_masked",
    "bureau",
    "legal_safe_summary",
    "inquiry_date",
}

MEDICAL_FIELDS = {
    "creditor_name",
    "account_number_masked",
    "bureau",
    "legal_safe_summary",
    "amount",
    "medical_status",
}

FRAUD_FIELDS = {
    "creditor_name",
    "account_number_masked",
    "bureau",
    "legal_safe_summary",
    "is_identity_theft",
    "fcra_605b",
    "ftc_report",
    "block_or_remove_request",
    "response_window",
}


@pytest.mark.parametrize(
    "tag,template,fields",
    [
        (
            "personal_info_correction",
            "personal_info_correction_letter_template.html",
            PII_FIELDS,
        ),
        (
            "inquiry_dispute",
            "inquiry_dispute_letter_template.html",
            INQUIRY_FIELDS,
        ),
        (
            "medical_dispute",
            "medical_dispute_letter_template.html",
            MEDICAL_FIELDS,
        ),
        ("fraud_dispute", "fraud_dispute_letter_template.html", FRAUD_FIELDS),
        ("custom_letter", "general_letter_template.html", {"recipient"}),
    ],
)
def test_finalize_routing_missing_fields(monkeypatch, tag, template, fields):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()

    decision = select_template(tag, {}, phase="finalize")

    assert decision.template_path == template
    assert set(decision.missing_fields) == fields

    counters = get_counters()
    assert counters.get("router.finalized") == 1
    assert counters.get(f"router.finalized.{tag}") == 1
    assert counters.get(f"router.finalized.{tag}.{template}") == 1
    for field in fields:
        key = f"router.missing_fields.{tag}.{template}.{field}"
        assert counters.get(key) == 1
        assert (
            counters.get(f"router.missing_fields.finalize.{tag}.{field}") == 1
        )


def test_instruction_skips_validation(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()

    decision = select_template("instruction", {}, phase="candidate")

    assert decision.template_path == "instruction_template.html"
    assert decision.missing_fields == []

    counters = get_counters()
    assert counters.get("router.candidate_selected") == 1
    assert counters.get("router.candidate_selected.instruction") == 1
    assert (
        counters.get(
            "router.candidate_selected.instruction.instruction_template.html"
        )
        == 1
    )
    assert not any(
        key.startswith("router.missing_fields.instruction") for key in counters
    )


def test_duplicate_emits_metrics(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()

    decision = select_template("duplicate", {}, phase="candidate")

    assert decision.template_path is None
    assert decision.missing_fields == []
    assert decision.router_mode == "memo"

    counters = get_counters()
    assert counters.get("router.skipped.duplicate") == 1
    assert counters.get("router.candidate_selected") == 1
    assert counters.get("router.candidate_selected.duplicate") == 1
    assert not any(
        key.startswith("router.candidate_selected.duplicate.")
        for key in counters
    )
