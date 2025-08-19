from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.core.letters.sanitizer import sanitize_rendered_html
from backend.core.logic.rendering.letter_rendering import render_dispute_letter_html


def test_sanitize_blocks_goodwill_for_collections():
    reset_counters()
    html = "<p>This is a goodwill adjustment request.</p>"
    ctx = {"accounts": [{"action_tag": "collection"}]}
    sanitized, overrides = sanitize_rendered_html(
        html, "dispute_letter_template.html", ctx
    )
    assert "goodwill" not in sanitized.lower()
    assert overrides == ["goodwill"]
    counters = get_counters()
    assert (
        counters["sanitizer.applied.dispute_letter_template.html"] == 1
    )
    assert (
        counters["policy_override_reason.dispute_letter_template.html.goodwill"]
        == 1
    )


def test_sanitize_noop_for_clean_html():
    reset_counters()
    html = "<p>All good here.</p>"
    sanitized, overrides = sanitize_rendered_html(
        html, "dispute_letter_template.html", {}
    )
    assert sanitized == html
    assert overrides == []
    counters = get_counters()
    assert (
        "sanitizer.applied.dispute_letter_template.html" not in counters
    )


def _base_ctx(body: str) -> dict:
    return {
        "client_name": "Jane Doe",
        "client_street": "1 Main St",
        "client_city": "Town",
        "client_state": "CA",
        "client_zip": "12345",
        "date": "Jan 1, 2024",
        "recipient_name": "Recipient",
        "greeting_line": "Hello",
        "body_paragraph": body,
    }


def test_render_dispute_letter_html_runs_sanitizer():
    reset_counters()
    ctx = _base_ctx("This includes a promise to pay statement.")
    artifact = render_dispute_letter_html(ctx, "general_letter_template.html")
    assert "promise to pay" not in artifact.html.lower()
    counters = get_counters()
    assert counters["sanitizer.applied.general_letter_template.html"] == 1


def test_render_dispute_letter_html_noop_for_clean_doc():
    reset_counters()
    ctx = _base_ctx("Just a friendly note.")
    artifact = render_dispute_letter_html(ctx, "general_letter_template.html")
    assert "friendly note" in artifact.html
    counters = get_counters()
    assert "sanitizer.applied.general_letter_template.html" not in counters
