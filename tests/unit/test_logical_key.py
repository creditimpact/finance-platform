from backend.core.logic.report_analysis.keys import normalize_issuer, compute_logical_account_key


def test_normalize_issuer_variants_collapse():
    assert normalize_issuer("U S  BANK.") == "US BANK"
    assert normalize_issuer("u.s-bank") == "US BANK"
    assert normalize_issuer("US BANK") == "US BANK"


def test_key_with_last4_includes_issuer_and_is_stable():
    k1 = compute_logical_account_key("U S BANK", "1234", "REVOLVING", "2020-01-01")
    k2 = compute_logical_account_key("US  BANK.", "1234", "REVOLVING", "2020-01-01")
    assert k1 == k2 and k1 is not None


def test_key_without_last4_fallback_works():
    k = compute_logical_account_key("US BANK", None, "REVOLVING", "2020-01-01")
    assert k is not None and len(k) == 16


def test_missing_everything_returns_none():
    assert compute_logical_account_key("", None, "", "") is None
