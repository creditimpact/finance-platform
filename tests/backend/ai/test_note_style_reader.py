import json
from pathlib import Path

from backend.ai.note_style_reader import get_style_metadata
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths


def _write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_get_style_metadata_returns_sanitized_payload(tmp_path: Path) -> None:
    sid = "SID-100"
    account_id = "idx-001"
    paths = ensure_note_style_paths(tmp_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    payload = {
        "analysis": {
            "tone": "Empathetic",
            "context_hints": {"topic": "payment_dispute"},
            "emphasis": ["paid_already", "support_request", "paid_already"],
        }
    }
    _write_result(account_paths.result_file, payload)

    metadata = get_style_metadata(sid, account_id, runs_root=tmp_path)
    assert metadata == {
        "tone": "Empathetic",
        "topic": "payment_dispute",
        "emphasis": ["paid_already", "support_request"],
    }


def test_get_style_metadata_missing_returns_none(tmp_path: Path) -> None:
    sid = "SID-200"
    account_id = "idx-002"

    metadata = get_style_metadata(sid, account_id, runs_root=tmp_path)
    assert metadata is None
