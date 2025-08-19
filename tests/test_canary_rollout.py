import os
import random

from backend.core.letters.router import _enabled
from backend.analytics.analytics_tracker import (
    reset_counters,
    emit_counter,
    set_metric,
    reset_ai_stats,
    check_canary_guardrails,
    get_counters,
    reset_canary_decisions,
    get_canary_decisions,
    log_ai_request,
)


def test_canary_toggle(monkeypatch):
    """Ensure ROUTER_CANARY_PERCENT controls routing proportion."""
    monkeypatch.setenv("ROUTER_RENDER_MS_P95_CEILING", "1000")
    monkeypatch.setenv("ROUTER_SANITIZER_RATE_CAP", "1")
    monkeypatch.setenv("ROUTER_AI_DAILY_BUDGET", "100000")

    monkeypatch.setenv("ROUTER_CANARY_PERCENT", "0")
    assert _enabled() is False

    monkeypatch.setenv("ROUTER_CANARY_PERCENT", "100")
    assert _enabled() is True

    monkeypatch.setenv("ROUTER_CANARY_PERCENT", "50")
    monkeypatch.setattr(random, "randint", lambda a, b: 25)
    assert _enabled() is True
    monkeypatch.setattr(random, "randint", lambda a, b: 75)
    assert _enabled() is False


def test_canary_halts_on_slo_breach(monkeypatch):
    reset_counters()
    reset_ai_stats()
    reset_canary_decisions()
    monkeypatch.setenv("ROUTER_CANARY_PERCENT", "50")

    emit_counter("router.finalized.dispute_letter_template.html", 100)
    emit_counter("validation.failed.dispute_letter_template.html", 5)
    set_metric("letter.render_ms.dispute_letter_template.html", 300)
    emit_counter("sanitizer.applied.dispute_letter_template.html", 10)
    log_ai_request(0, 0, 200, 0)

    breached = check_canary_guardrails(250, 0.05, 100)
    assert breached is True
    assert os.environ.get("ROUTER_CANARY_PERCENT") == "0"
    assert get_counters().get("canary.halt") == 1
    assert any(d["decision"] == "halt" for d in get_canary_decisions())
