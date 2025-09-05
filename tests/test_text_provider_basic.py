from backend.core.logic.report_analysis.text_provider import load_cached_text

def test_load_cached_text_returns_none():
    assert load_cached_text("unknown-session") is None
