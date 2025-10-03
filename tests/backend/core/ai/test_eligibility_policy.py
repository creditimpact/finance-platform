from backend.core.ai import eligibility_policy as policy


def test_policy_sets_have_expected_sizes():
    assert len(policy.ALWAYS_ELIGIBLE_FIELDS) == 18
    assert len(policy.CONDITIONAL_FIELDS) == 3
    assert policy.ALL_POLICY_FIELDS == policy.ALWAYS_ELIGIBLE_FIELDS | policy.CONDITIONAL_FIELDS
    assert len(policy.ALL_POLICY_FIELDS) == 21


def test_is_missing_values():
    assert policy.is_missing(None) is True
    assert policy.is_missing("--") is True
    assert policy.is_missing("   ") is True
    assert policy.is_missing(0) is False
    assert policy.is_missing("value") is False


def test_canonicalize_scalar_normalization():
    assert policy.canonicalize_scalar("  Foo  Bar  ") == "foo bar"
    assert policy.canonicalize_scalar("FOO,BAR") == "foo bar"
    assert policy.canonicalize_scalar("--") is None
    assert policy.canonicalize_scalar(None) is None
    assert policy.canonicalize_scalar("") is None
    assert policy.canonicalize_scalar("Value!") == "value"


def test_canonicalize_history_list():
    assert (
        policy.canonicalize_history(["  OK  ", " LATE "]) == "ok|late"
    )


def test_canonicalize_history_dict():
    history = {"late30": 3, "late90": 9, "late60": 2}
    assert policy.canonicalize_history(history) == "late30=3;late60=2;late90=9"


def test_canonicalize_history_fallback_scalar():
    assert policy.canonicalize_history("  Mixed Value ") == "mixed value"

