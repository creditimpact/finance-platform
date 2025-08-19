import os

import pytest
from jinja2 import Environment, FileSystemLoader

from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.assets.paths import templates_path
from backend.core.letters.client_context import (
    choose_phrase_template,
    format_safe_client_context,
    render_phrase,
)


def test_choose_phrase_template_neutral():
    tpl = choose_phrase_template("fraud_dispute", {}, [])
    assert tpl["template"].startswith("I am disputing")


def test_render_phrase_masks_pii():
    result = render_phrase("Account {account_ref}", {"account_ref": "123-45-6789"})
    assert "***-**-6789" in result


def test_format_safe_client_context_blocks_policy():
    reset_counters()
    sentence = format_safe_client_context(
        "goodwill", "", {"prohibited_admission_detected": True}, []
    )
    assert sentence is None
    counters = get_counters()
    assert counters["client_context_used.goodwill.neutral.policy_blocked"] == 1


def test_format_safe_client_context_ok():
    reset_counters()
    sentence = format_safe_client_context("debt_validation", "", {}, [])
    assert sentence
    assert len(sentence) <= 150
    counters = get_counters()
    assert counters["client_context_used.debt_validation.neutral.ok"] == 1


def test_format_safe_client_context_too_long(monkeypatch):
    import backend.core.letters.client_context as cc

    reset_counters()
    monkeypatch.setattr(
        cc,
        "choose_phrase_template",
        lambda *a, **k: {"key": "neutral", "template": "a" * 200},
    )
    sentence = cc.format_safe_client_context("pay_for_delete", "", {}, [])
    assert sentence is None
    counters = get_counters()
    assert counters["client_context_used.pay_for_delete.neutral.too_long"] == 1


def test_format_safe_client_context_banned(monkeypatch):
    import backend.core.letters.client_context as cc

    reset_counters()
    monkeypatch.setattr(
        cc,
        "choose_phrase_template",
        lambda *a, **k: {"key": "neutral", "template": "I promise to pay"},
    )
    sentence = cc.format_safe_client_context("debt_validation", "", {}, [])
    assert sentence is None
    counters = get_counters()
    assert counters["client_context_used.debt_validation.neutral.policy_blocked"] == 1


def _render_template(template_name: str, context: dict) -> str:
    env = Environment(loader=FileSystemLoader(templates_path("")))
    template = env.get_template(template_name)
    return template.render(**context)


@pytest.mark.parametrize(
    "tag, template, ctx",
    [
        (
            "debt_validation",
            "debt_validation_letter_template.html",
            {
                "collector_name": "Collector",
                "account_number_masked": "****1234",
                "bureau": "Experian",
                "legal_safe_summary": "Summary",
                "days_since_first_contact": "10",
            },
        ),
        (
            "fraud_dispute",
            "fraud_dispute_letter_template.html",
            {
                "creditor_name": "Creditor",
                "account_number_masked": "****1234",
                "bureau": "Experian",
                "legal_safe_summary": "Summary",
                "is_identity_theft": "Yes",
            },
        ),
        (
            "goodwill",
            "goodwill_letter_template.html",
            {
                "creditor_name": "Creditor",
                "account_number_masked": "****1234",
                "legal_safe_summary": "Summary",
                "months_since_last_late": "5",
                "account_history_good": "History",
            },
        ),
        (
            "pay_for_delete",
            "pay_for_delete_letter_template.html",
            {
                "collector_name": "Collector",
                "account_number_masked": "****1234",
                "legal_safe_summary": "Summary",
                "offer_terms": "Offer",
            },
        ),
    ],
)
def test_templates_include_sentence(tag, template, ctx):
    sentence = format_safe_client_context(tag, "", {}, [])
    ctx["client_context_sentence"] = sentence
    ctx.update({"client": type("C", (), {"full_name": "John Doe", "address_line": "123"})(), "today": "Jan 1, 2024"})
    html = _render_template(template, ctx)
    assert sentence in html
