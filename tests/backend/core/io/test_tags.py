from __future__ import annotations

import json
import os
from pathlib import Path

from backend.core.io.tags import read_tags, upsert_tag, write_tags_atomic


def _account_dir(tmp_path: Path) -> Path:
    account_dir = tmp_path / "accounts" / "123"
    account_dir.mkdir(parents=True, exist_ok=True)
    return account_dir


def test_upsert_tag_updates_without_duplicates(tmp_path: Path) -> None:
    account_dir = _account_dir(tmp_path)
    initial = {
        "kind": "merge_pair",
        "with": 42,
        "source": "merge_scorer",
        "score": 87,
    }

    upsert_tag(account_dir, initial, ("kind", "with"))

    update_payload = {
        "kind": "merge_pair",
        "with": 42,
        "source": "merge_scorer",
        "decision": "ai",
        "reason": "new_context",
    }

    upsert_tag(account_dir, update_payload, ("kind", "with"))

    tags = read_tags(account_dir)
    assert len(tags) == 1
    result = tags[0]
    assert result["decision"] == "ai"
    assert result["reason"] == "new_context"
    # ``score`` should persist even though it was not in the update payload.
    assert result["score"] == 87


def test_upsert_tag_collapses_existing_duplicates(tmp_path: Path) -> None:
    account_dir = _account_dir(tmp_path)
    duplicate_key = {"kind": "ai_decision", "with": 7, "source": "ai"}
    write_tags_atomic(
        account_dir,
        [
            {**duplicate_key, "decision": "merge", "reason": "first"},
            {**duplicate_key, "reason": "second"},
        ],
    )

    upsert_tag(
        account_dir,
        {**duplicate_key, "decision": "different", "at": "2024-06-01T00:00:00Z"},
    )

    tags = read_tags(account_dir)
    assert len(tags) == 1
    entry = tags[0]
    assert entry["decision"] == "different"
    assert entry["reason"] == "second"
    assert entry["at"] == "2024-06-01T00:00:00Z"


def test_write_tags_atomic_replaces_file(tmp_path: Path) -> None:
    account_dir = _account_dir(tmp_path)
    tags_path = account_dir / "tags.json"
    tags_path.write_text(json.dumps([{"kind": "initial"}]), encoding="utf-8")
    original_stat = tags_path.stat()

    write_tags_atomic(account_dir, [{"kind": "updated"}])

    new_stat = tags_path.stat()
    assert original_stat.st_ino != new_stat.st_ino
    assert read_tags(account_dir) == [{"kind": "updated"}]

    # Ensure the temporary file was removed.
    leftovers = [
        name for name in os.listdir(account_dir) if name.startswith("tmp") or name.endswith(".tmp")
    ]
    assert leftovers == []
