import os
import re
from pathlib import Path

import pytest

from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.core.letters import validators
from backend.core.letters.client_context import format_safe_client_context
from backend.core.letters.router import select_template
from backend.core.logic.rendering.letter_rendering import render_dispute_letter_html

BASE_CTX = {
    "client": {"full_name": "Jane Doe", "address_line": "123 Main St"},
    "today": "January 1, 2024",
    "account_number_masked": "1234",
    "bureau": "Experian",
}

SCENARIOS = [
    {
        "name": "fraud_with_ftc",
        "action_tag": "fraud_dispute",
        "initial_ctx": {
            "creditor_name": "Fraud Corp",
            "legal_safe_summary": "FCRA 605B allows me to block or remove this item via FTC report within 30 days.",
            "is_identity_theft": True,
        },
        "expect_html": True,
        "template": "fraud_dispute_letter_template.html",
        "golden": Path("tests/letters/goldens/fraud_with_ftc.html"),
    },
    {
        "name": "fraud_without_ftc",
        "action_tag": "fraud_dispute",
        "initial_ctx": {
            "creditor_name": "Fraud Corp",
            "legal_safe_summary": "FCRA 605B allows me to remove this item within 30 days.",
            "is_identity_theft": True,
        },
        "expect_html": False,
        "expect_validation_failure": True,
        "template": "fraud_dispute_letter_template.html",
    },
    {
        "name": "debt_validation",
        "action_tag": "debt_validation",
        "initial_ctx": {
            "legal_safe_summary": "FDCPA 1692g requires validation; please respond within 30 days.",
            "days_since_first_contact": "5",
        },
        "strategy_ctx": {"collector_name": "Collector Inc"},
        "expect_html": True,
        "template": "debt_validation_letter_template.html",
        "golden": Path("tests/letters/goldens/debt_validation.html"),
    },
    {
        "name": "mov",
        "action_tag": "mov",
        "initial_ctx": {
            "creditor_name": "Bank",
            "legal_safe_summary": "Please reinvestigate this item.",
            "cra_last_result": "verified",
            "days_since_cra_result": "45",
        },
        "expect_html": True,
        "template": "mov_letter_template.html",
        "golden": Path("tests/letters/goldens/mov.html"),
    },
    {
        "name": "pay_for_delete",
        "action_tag": "pay_for_delete",
        "initial_ctx": {
            "collector_name": "Collector Inc",
            "legal_safe_summary": "I will pay if you delete this account.",
            "offer_terms": "Delete account and we pay 60%.",
        },
        "expect_html": True,
        "template": "pay_for_delete_letter_template.html",
        "golden": Path("tests/letters/goldens/pay_for_delete.html"),
    },
    {
        "name": "cease_and_desist",
        "action_tag": "cease_and_desist",
        "initial_ctx": {
            "collector_name": "Collector Inc",
            "legal_safe_summary": "Stop contact immediately.",
        },
        "expect_html": True,
        "template": "cease_and_desist_letter_template.html",
        "golden": Path("tests/letters/goldens/cease_and_desist.html"),
    },
    {
        "name": "medical",
        "action_tag": "direct_dispute",
        "initial_ctx": {
            "creditor_name": "Medical Co",
            "legal_safe_summary": "Paid medical debt under $500 should be removed.",
            "furnisher_address": "1 Med St",
        },
        "expect_html": True,
        "template": "direct_dispute_letter_template.html",
        "golden": Path("tests/letters/goldens/medical.html"),
    },
    {
        "name": "unauthorized_inquiry",
        "action_tag": "direct_dispute",
        "initial_ctx": {
            "creditor_name": "Inquiry Co",
            "legal_safe_summary": "This unauthorized inquiry should be removed.",
            "furnisher_address": "2 Inquiry Ave",
        },
        "expect_html": True,
        "template": "direct_dispute_letter_template.html",
        "golden": Path("tests/letters/goldens/unauthorized_inquiry.html"),
    },
    {
        "name": "duplicate_tradeline",
        "action_tag": "direct_dispute",
        "initial_ctx": {
            "creditor_name": "Dup Co",
            "legal_safe_summary": "This tradeline appears duplicated.",
            "furnisher_address": "3 Dup Rd",
        },
        "expect_html": True,
        "template": "direct_dispute_letter_template.html",
        "golden": Path("tests/letters/goldens/duplicate_tradeline.html"),
    },
    {
        "name": "goodwill",
        "action_tag": "goodwill",
        "initial_ctx": {
            "creditor": "Good Bank",
            "creditor_name": "Good Bank",
            "legal_safe_summary": "I respectfully request a goodwill adjustment based on my positive payment history without admitting responsibility.",
            "months_since_last_late": "12",
            "account_history_good": "I have a positive history with your bank.",
        },
        "expect_html": True,
        "template": "goodwill_letter_template.html",
        "golden": Path("tests/letters/goldens/goodwill.html"),
    },
    {
        "name": "ignore",
        "action_tag": "ignore",
        "initial_ctx": {},
        "expect_html": False,
        "template": None,
    },
]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_letter_pipeline_golden(scenario):
    os.environ["LETTERS_ROUTER_PHASED"] = "1"
    reset_counters()
    ctx = BASE_CTX.copy()
    ctx.update(scenario.get("initial_ctx", {}))

    select_template(scenario["action_tag"], ctx, phase="candidate")

    if scenario.get("strategy_ctx"):
        ctx.update(scenario["strategy_ctx"])

    ctx["client_context_sentence"] = format_safe_client_context(
        scenario["action_tag"], "", {}, []
    )

    final_decision = select_template(scenario["action_tag"], ctx, phase="finalize")
    template = final_decision.template_path

    if template:
        missing = validators.validate_required_fields(
            template, ctx, final_decision.required_fields, validators.CHECKLIST
        )
    else:
        missing = []

    if scenario["expect_html"]:
        assert template == scenario["template"]
        assert not missing
        artifact = render_dispute_letter_html(ctx, template)
        html = re.sub(r"\s+", " ", artifact.html).strip()
        expected = re.sub(
            r"\s+", " ", scenario["golden"].read_text()
        ).strip()
        assert html == expected
    else:
        if scenario.get("expect_validation_failure"):
            assert template == scenario["template"]
            assert not missing
            with pytest.raises(ValueError):
                render_dispute_letter_html(ctx, template)
        else:
            assert template is None

    counters = get_counters()
    tag = scenario["action_tag"]
    if template:
        assert counters.get("router.candidate_selected") == 1
        assert counters.get(f"router.candidate_selected.{tag}") == 1
        assert ( 
            counters.get(
                f"router.candidate_selected.{tag}.{template}"
            )
            == 1
        )
        assert counters.get("router.finalized") == 1
        assert counters.get(f"router.finalized.{tag}") == 1
        assert counters.get(f"router.finalized.{tag}.{template}") == 1
    else:
        assert "router.candidate_selected" not in counters
        assert f"router.candidate_selected.{tag}" not in counters
        assert "router.finalized" not in counters
        assert f"router.finalized.{tag}" not in counters

    if scenario["expect_html"]:
        assert counters.get(f"letter_template_selected.{template}") == 1
        assert not any(k.startswith("validation.failed") for k in counters)
    else:
        if scenario.get("expect_validation_failure"):
            assert any(k.startswith("validation.failed") for k in counters)
            assert counters.get(f"letter_template_selected.{template}") is None
        else:
            assert not any(k.startswith("validation.failed") for k in counters)
            assert all(not k.startswith("letter_template_selected.") for k in counters)


def test_letter_pipeline_missing_fields():
    os.environ["LETTERS_ROUTER_PHASED"] = "1"
    reset_counters()
    ctx = BASE_CTX.copy()
    ctx.update(
        {
            "creditor_name": "Bank",
            "cra_last_result": "verified",
            "days_since_cra_result": "45",
        }
    )

    select_template("mov", ctx, phase="candidate")
    decision = select_template("mov", ctx, phase="finalize")

    assert decision.template_path == "default_dispute.html"
    assert "legal_safe_summary" in decision.missing_fields

    counters = get_counters()
    assert counters.get("router.finalize_errors") == 1


def test_letter_pipeline_render_error(monkeypatch):
    os.environ["LETTERS_ROUTER_PHASED"] = "1"
    reset_counters()
    ctx = BASE_CTX.copy()
    ctx.update(
        {
            "creditor_name": "Bank",
            "legal_safe_summary": "Please reinvestigate this item.",
            "cra_last_result": "verified",
            "days_since_cra_result": "45",
        }
    )

    select_template("mov", ctx, phase="candidate")

    class BrokenTemplate:
        def render(self, **_ctx):
            raise RuntimeError("boom")

    class BrokenEnv:
        def get_template(self, _name):
            return BrokenTemplate()

    import backend.core.letters.router as letters_router

    monkeypatch.setattr(letters_router, "Environment", lambda *_, **__: BrokenEnv())

    decision = select_template("mov", ctx, phase="finalize")

    assert decision.router_mode == "error"
    assert decision.template_path is None

    counters = get_counters()
    assert counters.get("router.render_error") == 1
    assert counters.get("router.finalized") is None
