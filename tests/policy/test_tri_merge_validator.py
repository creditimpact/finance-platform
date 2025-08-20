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
