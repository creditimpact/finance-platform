from backend.analytics.analytics_tracker import get_counters, reset_counters


def test_metrics_emission(monkeypatch, tmp_path):
    from backend.core.logic.report_analysis import report_prompting as rp

    # Single segment without chunking
    monkeypatch.setattr(rp.FLAGS, "chunk_by_bureau", False)
    monkeypatch.setattr(rp.FLAGS, "max_segment_tokens", 10_000)
    monkeypatch.setattr(rp, "USE_ANALYSIS_CACHE", True)

    # Fake analysis response with token usage and confidence
    def fake_analyze_bureau(*args, **kwargs):
        return {
            "summary_metrics": {
                "stage3_tokens_in": 5,
                "stage3_tokens_out": 7,
            },
            "confidence": 0.8,
            "needs_human_review": False,
        }, None

    monkeypatch.setattr(rp, "analyze_bureau", fake_analyze_bureau)
    from backend.core.logic.report_analysis import analysis_cache

    analysis_cache.reset_cache()
    monkeypatch.setattr(rp, "store_cached_analysis", lambda *a, **k: None)

    events = []
    monkeypatch.setattr(rp, "emit_event", lambda e, p: events.append((e, p)))

    reset_counters()

    rp.call_ai_analysis(
        text="Sample",
        is_identity_theft=False,
        output_json_path=tmp_path / "out.json",
        ai_client=None,
        strategic_context=None,
        request_id="req",
        doc_fingerprint="doc",
    )

    assert events and events[0][0] == "report_segment"
    payload = events[0][1]

    assert payload["stage3_tokens_in"] == 5
    assert payload["stage3_tokens_out"] == 7
    assert payload["cost_est"] > 0
    assert payload["cache_hit"] is False
    assert "confidence" in payload
    assert "needs_human_review" in payload
    assert "missing_bureaus" in payload
    assert payload["repair_count"] == 0
    assert payload["remediation_applied"] is False

    counters = get_counters()
    assert counters["analysis.cache_miss"] == 1
