from backend.policy.policy_loader import load_rulebook


def test_load_rulebook_resolves_placeholders_and_exposes_version() -> None:
    rulebook = load_rulebook()
    # Ensure version attribute exists
    assert isinstance(rulebook.version, str) and rulebook.version

    # Check that a known placeholder was resolved
    rule = next(r for r in rulebook["rules"] if r["id"] == "D_VALIDATION")
    cond = rule["when"]["all"][1]
    assert cond["lte"] == rulebook["limits"]["D_VALIDATION_WINDOW_DAYS"]
