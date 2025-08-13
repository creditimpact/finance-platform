from pathlib import Path
import hashlib

from backend.policy.policy_loader import get_rulebook_version, load_rulebook


def test_load_rulebook_empty_rules():
    data = load_rulebook()
    assert data["rules"] == []


def test_rulebook_version_matches_file_hash():
    version = get_rulebook_version()
    expected = hashlib.sha256(
        Path("backend/policy/rulebook.yaml").read_bytes()
    ).hexdigest()
    assert version == expected
