from backend.core.io.tags import read_tags, upsert_tag


def test_upsert_tag_is_idempotent(tmp_path):
    path = tmp_path / "tags.json"
    tag = {"kind": "merge_pair", "with": "acct-123", "score": 87}

    upsert_tag(path, tag, ["kind", "with"])
    upsert_tag(path, tag, ["kind", "with"])

    tags = read_tags(path)
    assert tags == [tag]


def test_upsert_tag_updates_existing_entry(tmp_path):
    path = tmp_path / "tags.json"
    original = {"kind": "merge_pair", "with": "acct-456", "score": 50}
    updated = {"kind": "merge_pair", "with": "acct-456", "score": 92}

    upsert_tag(path, original, ["kind", "with"])
    upsert_tag(path, updated, ["kind", "with"])

    tags = read_tags(path)
    assert tags == [updated]
