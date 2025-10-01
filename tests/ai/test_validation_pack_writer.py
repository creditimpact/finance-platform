from __future__ import annotations

import json
from pathlib import Path

from backend.ai.validation_builder import ValidationPackWriter


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_writer_builds_pack_lines(tmp_path: Path) -> None:
    sid = "S123"
    runs_root = tmp_path / "runs"

    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "balance_owed",
                    "category": "activity",
                    "min_days": 30,
                    "documents": ["statement", "   "],
                    "strength": "weak",
                    "ai_needed": True,
                },
                {
                    "field": "account_status",
                    "category": "status",
                    "ai_needed": False,
                },
            ],
            "field_consistency": {
                "balance_owed": {
                    "consensus": "split",
                    "disagreeing_bureaus": ["experian"],
                    "missing_bureaus": ["equifax"],
                    "history": {"2y": {"late_payments": 1}},
                    "raw": {
                        "transunion": "$100",
                        "experian": "$150",
                    },
                    "normalized": {
                        "transunion": 100,
                        "experian": 150,
                    },
                }
            },
        }
    }

    bureaus_payload = {
        "transunion": {"balance_owed": "$100"},
        "experian": {"balance_owed": "$155"},
        "equifax": {"balance_owed": None},
    }

    account_dir = runs_root / sid / "cases" / "accounts" / "1"
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    result = writer.write_all_packs()

    assert set(result) == {1}
    lines = result[1]
    assert len(lines) == 1

    payload = lines[0].payload
    assert payload["id"] == "acc_001__balance_owed"
    assert payload["field"] == "balance_owed"
    assert payload["strength"] == "weak"
    assert payload["documents"] == ["statement"]

    bureaus = payload["bureaus"]
    assert bureaus["transunion"]["raw"] == "$100"
    assert bureaus["experian"]["normalized"] == 150
    assert bureaus["equifax"]["raw"] is None

    context = payload["context"]
    assert context["consensus"] == "split"
    assert context["disagreeing_bureaus"] == ["experian"]
    assert context["missing_bureaus"] == ["equifax"]
    assert context["history"] == {"2y": {"late_payments": 1}}

    prompt = payload["prompt"]
    assert prompt["user"]["field_key"] == "balance_owed"
    assert "strong" in prompt["guidance"].lower()

    expected_output = payload["expected_output"]
    assert expected_output["properties"]["decision"]["enum"] == ["strong", "no_case"]

    pack_path = runs_root / sid / "ai_packs" / "validation" / "packs" / "val_acc_001.jsonl"
    on_disk = _read_jsonl(pack_path)
    assert len(on_disk) == 1
    assert on_disk[0]["id"] == payload["id"]


def test_writer_uses_bureau_fallback(tmp_path: Path) -> None:
    sid = "S234"
    runs_root = tmp_path / "runs"

    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "account_status",
                    "category": "status",
                    "strength": "soft",
                    "ai_needed": True,
                    "notes": "status mismatch",
                }
            ],
            "field_consistency": {},
        }
    }
    bureaus_payload = {
        "transunion": {"account_status": "Open"},
    }

    account_dir = runs_root / sid / "cases" / "accounts" / "7"
    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(7)

    assert len(lines) == 1
    payload = lines[0].payload
    assert payload["strength"] == "weak"
    assert payload["bureaus"]["transunion"]["raw"] == "Open"
    assert payload["context"]["requirement_note"] == "status mismatch"

    pack_path = runs_root / sid / "ai_packs" / "validation" / "packs" / "val_acc_007.jsonl"
    on_disk = _read_jsonl(pack_path)
    assert on_disk[0]["bureaus"]["transunion"]["raw"] == "Open"


def test_writer_skips_strong_fields(tmp_path: Path) -> None:
    sid = "S345"
    runs_root = tmp_path / "runs"

    summary_payload = {
        "validation_requirements": {
            "requirements": [
                {
                    "field": "payment_status",
                    "category": "status",
                    "strength": "strong",
                    "ai_needed": True,
                }
            ],
            "field_consistency": {},
        }
    }

    account_dir = runs_root / sid / "cases" / "accounts" / "9"
    _write_json(account_dir / "summary.json", summary_payload)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(9)

    assert lines == []

    pack_path = runs_root / sid / "ai_packs" / "validation" / "packs" / "val_acc_009.jsonl"
    assert pack_path.exists()
    assert pack_path.read_text(encoding="utf-8") == ""

