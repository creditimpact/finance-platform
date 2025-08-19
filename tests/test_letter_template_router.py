from backend.core.letters.router import select_template


def test_router_basic_mappings(monkeypatch):
    monkeypatch.setenv("LETTERS_ROUTER_ENABLED", "1")
    assert select_template("dispute", {}).template_path == "dispute_letter_template.html"
    assert select_template("goodwill", {}).template_path == "goodwill_letter_template.html"
    assert select_template("custom_letter", {}).template_path == "general_letter_template.html"
    decision = select_template("ignore", {})
    assert decision.template_path is None
    assert decision.router_mode == "skip"
