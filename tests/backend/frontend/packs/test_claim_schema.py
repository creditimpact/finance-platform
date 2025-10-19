import pytest

from backend.frontend.packs.claim_schema import resolve_issue_claims


@pytest.mark.parametrize(
    "issue,expected_keys",
    [
        (
            "collection",
            {
                "not_mine",
                "paid_in_full",
                "settled",
                "authorized_user",
                "mixed_file",
            },
        ),
        (
            "delinquency",
            {
                "never_late_bank_error",
                "goodwill",
                "wrong_late_code",
                "mixed_file",
            },
        ),
    ],
)
def test_resolve_issue_claims_includes_expected_claims(issue, expected_keys):
    base_docs, claims = resolve_issue_claims(issue)

    assert tuple(base_docs) == ("gov_id", "proof_of_address")

    keys = {claim.key for claim in claims}
    for key in expected_keys:
        assert key in keys


def test_resolve_issue_claims_collection_has_required_docs():
    _, claims = resolve_issue_claims("collection")
    by_key = {claim.key: claim for claim in claims}

    paid_in_full = by_key["paid_in_full"]
    assert tuple(paid_in_full.requires) == ("paid_in_full_letter", "proof_of_payment")

    not_mine = by_key["not_mine"]
    assert tuple(not_mine.requires) == ("ftc_id_theft_report",)


def test_resolve_issue_claims_delinquency_has_required_docs():
    _, claims = resolve_issue_claims("delinquency")
    by_key = {claim.key: claim for claim in claims}

    never_late = by_key["never_late_bank_error"]
    assert set(never_late.requires) == {"bank_statement", "billing_statement"}

    goodwill = by_key["goodwill"]
    assert tuple(goodwill.requires) == ("goodwill_support",)
