from pathlib import Path

from backend.core.logic.report_analysis import report_prompting as rp
from backend.core.logic.report_analysis.flags import FLAGS


class DummyAIClient:
    pass


def test_large_bureau_chunking(monkeypatch, tmp_path: Path):
    FLAGS.max_segment_tokens = 100
    calls = []

    def fake_analyze_bureau(text, **kwargs):
        calls.append(text)
        data = {
            "negative_accounts": [],
            "open_accounts_with_issues": [],
            "positive_accounts": [],
            "high_utilization_accounts": [],
            "all_accounts": [],
            "inquiries": [],
            "personal_info_issues": [],
            "account_inquiry_matches": [],
            "strategic_recommendations": [],
        }
        upper = text.upper()
        if "INQUIRIES" in upper:
            data["inquiries"] = [{"creditor_name": "Inq1", "bureau": "Experian"}]
        elif "COLLECTIONS" in upper:
            data["all_accounts"] = [{"name": "Collection", "bureau": "Experian"}]
        else:
            data["all_accounts"] = [{"name": "Account", "bureau": "Experian"}]
        return data, None

    monkeypatch.setattr(rp, "analyze_bureau", fake_analyze_bureau)

    text = (
        "EXPERIAN REPORT\n"
        + "ACCOUNTS\n" + "A " * 120
        + "\nCOLLECTIONS\n" + "B " * 120
        + "\nINQUIRIES\n" + "C " * 120
    )

    result = rp.call_ai_analysis(
        text=text,
        is_identity_theft=False,
        output_json_path=tmp_path / "out.json",
        ai_client=DummyAIClient(),
        strategic_context=None,
        request_id="req",
        doc_fingerprint="fingerprint",
    )

    names = [acc["name"] for acc in result["all_accounts"]]
    assert names == ["Account", "Collection"]
    assert result["inquiries"][0]["creditor_name"] == "Inq1"
    assert len(calls) == 3
