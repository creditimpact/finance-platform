import pytest

from backend.policy import policy_loader, validators


def _reset_cache() -> None:
    policy_loader._RULEBOOK_CACHE = None


def test_validator_detects_missing_rule(monkeypatch) -> None:
    _reset_cache()
    rulebook = policy_loader.load_rulebook()
    monkeypatch.setattr(
        validators,
        "TRI_MERGE_MISMATCH_TYPES",
        validators.TRI_MERGE_MISMATCH_TYPES | {"bogus"},
    )
    with pytest.raises(ValueError):
        validators.validate_tri_merge_mismatch_rules(rulebook)
    _reset_cache()


def test_rulebook_has_rules_for_all_tri_merge_mismatches() -> None:
    _reset_cache()
    rulebook = policy_loader.load_rulebook()
    validators.validate_tri_merge_mismatch_rules(rulebook)
    _reset_cache()
