from backend.core.letters.router import select_template


def test_router_basic_mappings(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    assert (
        select_template("dispute", {"bureau": "Experian"}, phase="finalize").template_path
        == "dispute_letter_template.html"
    )
    assert (
        select_template("goodwill", {"creditor": "ABC"}, phase="finalize").template_path
        == "goodwill_letter_template.html"
    )
    assert (
        select_template("custom_letter", {"recipient": "Joe"}, phase="finalize").template_path
        == "general_letter_template.html"
    )
    assert (
        select_template(
            "bureau_dispute",
            {
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
                "client_name": "Jane",
                "date": "2024-01-01",
                "accounts_summary": "summary",
                "per_account_actions": [
                    {"account_ref": "1", "action_sentence": "Pay down balance"}
                ],
            },
            phase="finalize",
        ).template_path
        == "instruction_template.html"
    )
    decision = select_template("ignore", {}, phase="finalize")
    assert decision.template_path is None
    assert decision.router_mode == "skip"


def test_missing_fields(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    decision = select_template("goodwill", {}, phase="candidate")
    assert decision.missing_fields == ["creditor"]
    decision = select_template("goodwill", {"creditor": "XYZ"}, phase="candidate")
    assert decision.missing_fields == []
    decision = select_template("bureau_dispute", {}, phase="candidate")
    assert decision.missing_fields == [
        "creditor_name",
        "account_number_masked",
        "bureau",
        "legal_safe_summary",
    ]
    decision = select_template(
        "bureau_dispute",
        {
            "creditor_name": "Creditor",
            "account_number_masked": "1234",
            "bureau": "Experian",
            "legal_safe_summary": "summary",
        },
        phase="candidate",
    )
    assert decision.missing_fields == []
