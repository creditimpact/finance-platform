import difflib

from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.core.letters.sanitizer import sanitize_rendered_html


def _diff(a: str, b: str) -> str:
    """Return a unified diff between ``a`` and ``b``."""
    return "\n".join(difflib.unified_diff(a.split(), b.split(), lineterm=""))


def test_sanitizer_redacts_and_normalizes_whitespace():
    reset_counters()
    html = (
        "<p>I promise to pay you.</p>  \n"
        "<p>Contact me at test@example.com   123-456-7890.</p>"
    )
    sanitized, overrides = sanitize_rendered_html(
        html, "dispute_letter_template.html", {}
    )
    diff = _diff(html, sanitized)
    assert diff, "expected sanitized output to differ"
    assert "[REDACTED]" in sanitized
    assert "promise to pay" not in sanitized.lower()
    assert set(overrides) >= {"whitespace", "pii", "promise to pay"}
    counters = get_counters()
    assert counters["sanitizer.success.dispute_letter_template.html"] == 1
    assert counters["sanitizer.applied.dispute_letter_template.html"] == 1
    assert "sanitizer.failure.dispute_letter_template.html" not in counters
    assert (
        counters["router.sanitize_success.dispute_letter_template.html"] == 1
    )
    assert "router.sanitize_failure.dispute_letter_template.html" not in counters


def test_sanitizer_reports_format_failure():
    reset_counters()
    html = "This text lacks required tags"
    sanitized, overrides = sanitize_rendered_html(
        html, "dispute_letter_template.html", {}
    )
    diff = _diff(html, sanitized)
    assert not diff, "unexpected diff for format failure"
    assert "format" in overrides
    counters = get_counters()
    assert counters["sanitizer.failure.dispute_letter_template.html"] == 1
    assert counters["sanitizer.applied.dispute_letter_template.html"] == 1
    assert "sanitizer.success.dispute_letter_template.html" not in counters
    assert (
        counters["router.sanitize_failure.dispute_letter_template.html"] == 1
    )
    assert "router.sanitize_success.dispute_letter_template.html" not in counters
