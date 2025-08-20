import pytest
from backend.core.letters.router import select_template


def _ctx(bureau: str) -> dict:
    return {
        "creditor_name": "Creditor",
        "account_number_masked": "1234",
        "bureau": bureau,
        "legal_safe_summary": "summary",
    }


def test_finalize_prefers_cra_specific_template(monkeypatch, tmp_path):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    from backend.core.letters import router as router_mod
    router_mod._ROUTER_CACHE.clear()
    (tmp_path / "experian_bureau_dispute_letter_template.html").write_text("dummy")
    monkeypatch.setattr(router_mod, "TEMPLATES_DIRS", [tmp_path])

    decision = select_template("bureau_dispute", _ctx("Experian"), phase="finalize")

    assert decision.template_path == "experian_bureau_dispute_letter_template.html"


def test_finalize_falls_back_to_generic(monkeypatch, tmp_path):
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")
    from backend.core.letters import router as router_mod
    router_mod._ROUTER_CACHE.clear()
    (tmp_path / "bureau_dispute_letter_template.html").write_text("dummy")
    monkeypatch.setattr(router_mod, "TEMPLATES_DIRS", [tmp_path])

    decision = select_template("bureau_dispute", _ctx("Equifax"), phase="finalize")

    assert decision.template_path == "bureau_dispute_letter_template.html"
