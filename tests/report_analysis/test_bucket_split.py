from pathlib import Path

import backend.core.logic.report_analysis.analyze_report as ar


def test_closed_derogatory_goes_negative():
    acc = {
        "name": "Bad Bank",
        "account_status": "Closed",
        "payment_status": "Derogatory",
        "closed_date": "2024-01-01",
        "issue_types": ["late_payment"],
        "primary_issue": "late_payment",
    }
    negatives, open_issues = ar._split_account_buckets([acc])
    assert [a["name"] for a in negatives] == ["Bad Bank"]
    assert not open_issues


def test_open_late_payments_go_open_issues():
    acc = {
        "name": "Good Bank",
        "account_status": "Open",
        "payment_status": "Open",
        "late_payments": {"30": 1},
        "issue_types": ["late_payment"],
        "primary_issue": "late_payment",
    }
    negatives, open_issues = ar._split_account_buckets([acc])
    assert not negatives
    assert [a["name"] for a in open_issues] == ["Good Bank"]


def test_split_accounts_integration(monkeypatch, tmp_path: Path):
    from backend.core.logic.report_analysis import analyze_report

    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_text("dummy", encoding="utf-8")
    out_path = tmp_path / "out.json"

    monkeypatch.setattr(analyze_report, "extract_text_from_pdf", lambda p: "text")
    monkeypatch.setattr(analyze_report, "extract_pdf_page_texts", lambda p: [])
    monkeypatch.setattr(
        analyze_report, "extract_late_history_blocks", lambda *a, **k: ({}, {}, {})
    )
    monkeypatch.setattr(
        analyze_report,
        "extract_three_column_fields",
        lambda p: ({}, {}, {}, {}, {}, {}, {}),
    )
    monkeypatch.setattr(
        analyze_report, "extract_payment_statuses", lambda text: ({}, {})
    )
    monkeypatch.setattr(analyze_report, "extract_creditor_remarks", lambda text: {})
    monkeypatch.setattr(analyze_report, "extract_account_numbers", lambda text: {})

    def fake_call_ai_analysis(*args, **kwargs):
        return {
            "all_accounts": [
                {
                    "name": "Bad Bank",
                    "account_status": "Closed",
                    "payment_status": "Derogatory",
                    "closed_date": "2024-01-01",
                    "issue_types": ["late_payment"],
                    "primary_issue": "late_payment",
                },
                {
                    "name": "Good Bank",
                    "account_status": "Open",
                    "payment_status": "Open",
                    "late_payments": {"30": 1},
                    "past_due_amount": 100,
                    "issue_types": ["late_payment"],
                    "primary_issue": "late_payment",
                },
            ],
            "negative_accounts": [],
            "open_accounts_with_issues": [],
            "positive_accounts": [],
            "high_utilization_accounts": [],
            "inquiries": [],
            "account_inquiry_matches": [],
            "summary_metrics": {},
            "strategic_recommendations": [],
        }

    monkeypatch.setattr(analyze_report, "call_ai_analysis", fake_call_ai_analysis)

    result = analyze_report.analyze_credit_report(
        pdf_path, out_path, {}, ai_client=None, run_ai=True, request_id="req"
    )
    assert [a["name"] for a in result["negative_accounts"]] == ["Bad Bank"]
    assert [a["name"] for a in result["open_accounts_with_issues"]] == ["Good Bank"]
