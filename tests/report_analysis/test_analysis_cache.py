import importlib
import sys
from pathlib import Path

# Ensure repo root is on path for direct imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.analytics import analytics_tracker
from tests.helpers.fake_ai_client import FakeAIClient


def test_analysis_cache_hits_and_invalidates(tmp_path, monkeypatch):
    """Repeated calls hit cache; prompt or schema changes invalidate."""

    report_prompting = importlib.import_module(
        "backend.core.logic.report_analysis.report_prompting"
    )
    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()
    analytics_tracker.reset_counters()

    client = FakeAIClient()
    client.add_chat_response('{"inquiries": [], "all_accounts": []}')
    out = tmp_path / "result.json"

    # Prime cache
    report_prompting.call_ai_analysis(
        "text",
        False,
        out,
        ai_client=client,
        strategic_context="goal",
        request_id="req",
        doc_fingerprint="fp",
    )

    # Second call with identical inputs should hit cache
    report_prompting.call_ai_analysis(
        "text",
        False,
        out,
        ai_client=client,
        strategic_context="goal",
        request_id="req",
        doc_fingerprint="fp",
    )

    counters = analytics_tracker.get_counters()
    assert counters.get("analysis.cache_hit") == 1
    assert len(client.chat_payloads) == 1

    # Changing prompt invalidates cache
    client.add_chat_response('{"inquiries": [], "all_accounts": []}')
    report_prompting.call_ai_analysis(
        "text",
        False,
        out,
        ai_client=client,
        strategic_context="other",
        request_id="req",
        doc_fingerprint="fp",
    )
    assert len(client.chat_payloads) == 2

    # Bumping schema invalidates cache
    monkeypatch.setattr(
        report_prompting,
        "ANALYSIS_SCHEMA_VERSION",
        report_prompting.ANALYSIS_SCHEMA_VERSION + 1,
    )
    client.add_chat_response('{"inquiries": [], "all_accounts": []}')
    report_prompting.call_ai_analysis(
        "text",
        False,
        out,
        ai_client=client,
        strategic_context="goal",
        request_id="req",
        doc_fingerprint="fp",
    )
    assert len(client.chat_payloads) == 3

    # Cache hit metric should still reflect only the first hit
    counters = analytics_tracker.get_counters()
    assert counters.get("analysis.cache_hit") == 1
