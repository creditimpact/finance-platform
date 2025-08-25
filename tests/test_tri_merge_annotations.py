import json

from backend.core.orchestrators import _annotate_with_tri_merge


def test_tri_merge_annotations_do_not_change_counts_and_primary_issue(monkeypatch):
    sections = {
        "negative_accounts": [
            {
                "account_id": "1",
                "name": "Cap One",
                "account_number": "1234",
                "bureaus": ["Experian"],
                "issue_types": ["late_payment"],
                "primary_issue": "late_payment",
            }
        ],
        "open_accounts_with_issues": [
            {
                "account_id": "2",
                "name": "Cap One",
                "account_number": "1234",
                "bureaus": ["Equifax"],
                "issue_types": ["late_payment"],
                "primary_issue": "late_payment",
            }
        ],
    }
    neg_before = len(sections["negative_accounts"])
    open_before = len(sections["open_accounts_with_issues"])
    monkeypatch.setenv("ENABLE_TRI_MERGE", "1")
    _annotate_with_tri_merge(sections)
    assert len(sections["negative_accounts"]) == neg_before
    assert len(sections["open_accounts_with_issues"]) == open_before
    assert "tri_merge" in sections["negative_accounts"][0]
    assert "tri_merge" in sections["open_accounts_with_issues"][0]
    assert sections["negative_accounts"][0]["primary_issue"] == "late_payment"
    assert sections["open_accounts_with_issues"][0]["primary_issue"] == "late_payment"


def test_tri_merge_violation_logged_and_reverted(monkeypatch):
    sections = {
        "negative_accounts": [
            {
                "account_id": "1",
                "name": "Cap One",
                "account_number": "1234",
                "bureaus": ["Experian"],
                "issue_types": ["late_payment"],
                "primary_issue": "late_payment",
            }
        ]
    }

    def bad_compute(families):
        sections["negative_accounts"][0]["primary_issue"] = "collection"
        return families

    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "backend.core.logic.report_analysis.tri_merge.compute_mismatches",
        bad_compute,
    )
    monkeypatch.setattr(
        "backend.audit.audit.emit_event",
        lambda name, payload, **kw: events.append((name, json.dumps(payload))),
    )
    monkeypatch.setenv("ENABLE_TRI_MERGE", "1")
    _annotate_with_tri_merge(sections)

    # Primary issue should be restored and violation logged
    assert sections["negative_accounts"][0]["primary_issue"] == "late_payment"
    assert any(evt[0] == "trimerge_violation" for evt in events)
