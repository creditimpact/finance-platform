from backend.core.orchestrators import _annotate_with_tri_merge


def test_tri_merge_annotations_do_not_change_counts(monkeypatch):
    sections = {
        "negative_accounts": [
            {
                "account_id": "1",
                "name": "Cap One",
                "account_number": "1234",
                "bureaus": ["Experian"],
                "issue_types": ["late_payment"],
            }
        ],
        "open_accounts_with_issues": [
            {
                "account_id": "2",
                "name": "Cap One",
                "account_number": "1234",
                "bureaus": ["Equifax"],
                "issue_types": ["late_payment"],
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
