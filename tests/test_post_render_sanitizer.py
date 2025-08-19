from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.core.letters.sanitizer import sanitize_rendered_html


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
