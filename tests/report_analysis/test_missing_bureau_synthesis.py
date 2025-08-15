import types
from pathlib import Path

import backend.core.logic.report_analysis.report_prompting as rp
from backend.core.logic.utils.names_normalization import BUREAUS


def test_missing_bureau_sections(monkeypatch, tmp_path: Path):
    """Missing bureaus should still emit placeholder sections."""
    # simplify prompt generation and caching
    monkeypatch.setattr(rp, "_generate_prompt", lambda *a, **k: ("", "", ""))
    monkeypatch.setattr(rp, "get_cached_analysis", lambda *a, **k: None)
    monkeypatch.setattr(rp, "store_cached_analysis", lambda *a, **k: None)

    def fake_analyze_bureau(text, **kwargs):
        return {"all_accounts": [{"name": "Cap One", "bureaus": ["Experian"]}], "inquiries": []}, None

    monkeypatch.setattr(rp, "analyze_bureau", fake_analyze_bureau)

    text = "Experian report\naccount details"

    result = rp.call_ai_analysis(
        text=text,
        is_identity_theft=False,
        output_json_path=tmp_path / "out.json",
        ai_client=types.SimpleNamespace(),
        strategic_context=None,
        request_id="req",
        doc_fingerprint="fp",
    )

    sections = {s["bureau"]: s for s in result["bureau_sections"]}
    assert set(sections.keys()) == set(BUREAUS)
    # Experian section comes from analysis and should not be marked missing
    assert not sections["Experian"]["is_missing"]
    for bureau in BUREAUS:
        if bureau == "Experian":
            continue
        placeholder = sections[bureau]
        assert placeholder["is_missing"]
        assert placeholder["all_accounts"] == []
        assert placeholder["inquiries"] == []
