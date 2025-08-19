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
    decision = select_template("ignore", {}, phase="finalize")
    assert decision.template_path is None
    assert decision.router_mode == "skip"


def test_missing_fields(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    decision = select_template("goodwill", {}, phase="candidate")
    assert decision.missing_fields == ["creditor"]
    decision = select_template("goodwill", {"creditor": "XYZ"}, phase="candidate")
    assert decision.missing_fields == []
