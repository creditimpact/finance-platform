import pytest

from backend.policy import policy_loader, validators


def _reset_cache() -> None:
    policy_loader._RULEBOOK_CACHE = None


def test_load_rulebook_resolves_placeholders_and_exposes_version() -> None:
    _reset_cache()
    rulebook = policy_loader.load_rulebook()
    # Ensure version attribute exists
    assert isinstance(rulebook.version, str) and rulebook.version

    # Check that a known placeholder was resolved
    rule = next(r for r in rulebook["rules"] if r["id"] == "D_VALIDATION")
    cond = rule["when"]["all"][1]
    assert cond["lte"] == rulebook["limits"]["D_VALIDATION_WINDOW_DAYS"]


def test_load_rulebook_fails_when_mismatch_type_missing(monkeypatch) -> None:
    _reset_cache()
    monkeypatch.setattr(
        validators,
        "TRI_MERGE_MISMATCH_TYPES",
        validators.TRI_MERGE_MISMATCH_TYPES | {"bogus"},
    )
    with pytest.raises(ValueError):
        policy_loader.load_rulebook()
