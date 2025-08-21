import pytest

from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.core.letters.router import select_template


def test_router_basic_mappings(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()
    base = {"client": {"full_name": "Jane", "address_line": "1 St"}, "today": "2024-01-01"}
    assert (
        select_template(
            "dispute", {**base, "bureau": "Experian"}, phase="finalize"
        ).template_path
        == "dispute_letter_template.html"
    )
    assert (
        select_template(
            "goodwill", {**base, "creditor": "ABC"}, phase="finalize"
        ).template_path
        == "goodwill_letter_template.html"
    )
    assert (
        select_template(
            "custom_letter", {**base, "recipient": "Joe"}, phase="finalize"
        ).template_path
        == "general_letter_template.html"
    )
    assert (
        select_template(
            "bureau_dispute",
            {
                **base,
                "creditor_name": "Creditor",
                "account_number_masked": "1234",
                "bureau": "Experian",
                "legal_safe_summary": "summary",
            },
            phase="finalize",
        ).template_path
        == "bureau_dispute_letter_template.html"
    )
    assert (
        select_template(
            "inquiry_dispute",
            {
                **base,
                "inquiry_creditor_name": "Creditor",
                "account_number_masked": "1234",
                "bureau": "Experian",
                "legal_safe_summary": "summary",
                "inquiry_date": "2024-01-01",
            },
            phase="finalize",
        ).template_path
        == "inquiry_dispute_letter_template.html"
    )
    assert (
        select_template(
            "medical_dispute",
            {
                **base,
                "creditor_name": "Creditor",
                "account_number_masked": "1234",
                "bureau": "Experian",
                "legal_safe_summary": "summary",
                "amount": "100",
                "medical_status": "paid",
            },
            phase="finalize",
        ).template_path
        == "medical_dispute_letter_template.html"
    )
    assert (
        select_template(
            "paydown_first",
            {
                **base,
                "client_name": "Jane",
                "date": "2024-01-01",
                "accounts_summary": {"loans": []},
                "per_account_actions": [
                    {"account_ref": "1", "action_sentence": "Pay down balance"}
                ],
            },
            phase="finalize",
        ).template_path
        == "instruction_template.html"
    )
    decision = select_template("ignore", base, phase="finalize")
    assert decision.template_path is None
    assert decision.router_mode == "skip"

    counters = get_counters()
    assert counters.get("router.skipped.paydown_first") == 1
    assert counters.get("router.skipped.ignore") == 1


def test_missing_fields(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    decision = select_template("goodwill", {}, phase="finalize")
    assert set(decision.missing_fields) == {
        "creditor",
        "non_promissory_tone",
        "positive_history_reference",
        "discretionary_request",
        "no_admission",
    }
    decision = select_template(
        "goodwill",
        {
            "creditor": "XYZ",
            "legal_safe_summary": "goodwill positive request without admit",
        },
        phase="finalize",
    )
    assert decision.missing_fields == []
    decision = select_template("bureau_dispute", {}, phase="finalize")
    assert set(decision.missing_fields) == {
        "creditor_name",
        "account_number_masked",
        "bureau",
        "legal_safe_summary",
        "fcra_611",
        "reinvestigation_request",
    }
    decision = select_template(
        "bureau_dispute",
        {
            "creditor_name": "Creditor",
            "account_number_masked": "1234",
            "bureau": "Experian",
            "legal_safe_summary": "Please reinvestigate under section 611",
        },
        phase="finalize",
    )
    assert decision.missing_fields == []


def test_unknown_action_tag_raises(monkeypatch, caplog):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError) as exc:
            select_template("bogus", {}, phase="candidate", session_id="sess1")
    assert "bogus" in str(exc.value)
    assert "sess1" in str(exc.value)
    counters = get_counters()
    assert counters.get("router.candidate_errors") == 1
    assert any("bogus" in r.message for r in caplog.records)


def test_unknown_template_name_raises(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    reset_counters()
    from backend.core.letters import router as router_mod

    monkeypatch.setattr(router_mod, "TEMPLATES_DIRS", [tmp_path])
    with caplog.at_level("ERROR"):
        with pytest.raises(ValueError) as exc:
            select_template(
                "dispute",
                {"bureau": "Experian"},
                phase="candidate",
                session_id="sess2",
            )
    assert "dispute_letter_template.html" in str(exc.value)
    assert "sess2" in str(exc.value)
    counters = get_counters()
    assert counters.get("router.candidate_errors") == 1
    assert any("dispute_letter_template.html" in r.message for r in caplog.records)


def test_routing_cache(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    from backend.core.letters import router as router_mod

    router_mod._ROUTER_CACHE.clear()
    decision1 = select_template(
        "goodwill", {"creditor": "ABC"}, phase="candidate", session_id="sess3"
    )
    decision2 = select_template(
        "goodwill", {}, phase="candidate", session_id="sess3"
    )
    assert decision1.missing_fields == []
    assert decision2.missing_fields == []
    assert decision1 is decision2
