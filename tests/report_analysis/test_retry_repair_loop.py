import logging


def test_retry_repair_loop(monkeypatch, caplog, tmp_path):
    from backend.core.logic.report_analysis import report_prompting as rp

    # Force single segment and generous token limit
    monkeypatch.setattr(rp.FLAGS, "chunk_by_bureau", False)
    monkeypatch.setattr(rp.FLAGS, "max_segment_tokens", 10_000)

    # Simulate headings so missing account issue is detected
    monkeypatch.setattr(
        rp,
        "extract_account_headings",
        lambda text: [("acme", "Acme Bank")],
    )

    calls = {"count": 0}

    def fake_analyze_bureau(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            # First pass returns a merged account triggering remediation
            return {"all_accounts": [{"name": "Acme Bank", "bureaus": []}]}, None
        return {"all_accounts": [{"name": "Acme Bank", "bureaus": ["Full"]}]}, None

    monkeypatch.setattr(rp, "analyze_bureau", fake_analyze_bureau)
    monkeypatch.setattr(rp, "get_cached_analysis", lambda *a, **k: None)
    monkeypatch.setattr(rp, "store_cached_analysis", lambda *a, **k: None)

    caplog.set_level(logging.INFO)

    result = rp.call_ai_analysis(
        text="Acme Bank\nDetails",
        is_identity_theft=False,
        output_json_path=tmp_path / "out.json",
        ai_client=None,
        strategic_context=None,
        request_id="req",
        doc_fingerprint="doc",
    )

    # Ensure remediation was attempted
    assert calls["count"] == 2
    assert any(
        r.message == "analysis_remediation" and getattr(r, "repair_step", 0) == 1
        for r in caplog.records
    )

    # Final result contains the recovered account
    assert result["all_accounts"][0]["name"] == "Acme Bank"
