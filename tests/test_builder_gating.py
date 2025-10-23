import json
from pathlib import Path

from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths
from backend.validation.build_packs import (
    ValidationPackBuilder,
    _PROMPT_USER_TEMPLATE,
    _SYSTEM_PROMPT,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _make_manifest(tmp_path: Path, sid: str = "SID") -> tuple[dict, Path, Path, Path]:
    base = tmp_path / sid
    accounts_dir = base / "cases" / "accounts"
    packs_dir = base / "ai_packs" / "validation" / "packs"
    results_dir = base / "ai_packs" / "validation" / "results"
    index_path = base / "ai_packs" / "validation" / "index.json"
    log_path = base / "ai_packs" / "validation" / "validation.log"

    manifest = {
        "sid": sid,
        "base_dirs": {"cases_accounts_dir": str(accounts_dir)},
        "ai": {
            "packs": {
                "validation": {
                    "packs_dir": str(packs_dir),
                    "results_dir": str(results_dir),
                    "index": str(index_path),
                    "logs": str(log_path),
                }
            }
        },
    }
    return manifest, accounts_dir, packs_dir, index_path


def test_builder_skips_accounts_without_eligible_findings(tmp_path: Path) -> None:
    manifest, accounts_dir, packs_dir, _ = _make_manifest(tmp_path)

    account_dir = accounts_dir / "1"
    account_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_type",
                    "strength": "weak",
                    "send_to_ai": False,
                    "reason_code": "C4_TWO_MATCH_ONE_DIFF",
                },
                {
                    "field": "creditor_type",
                    "strength": "weak",
                    "send_to_ai": False,
                    "reason_code": "C5_ALL_DIFF",
                },
            ],
            "field_consistency": {},
        }
    }

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", {})

    builder = ValidationPackBuilder(manifest)
    builder.build()

    pack_path = packs_dir / "val_acc_001.jsonl"
    assert not pack_path.exists()


def test_builder_writes_single_pack_with_two_lines(tmp_path: Path) -> None:
    manifest, accounts_dir, packs_dir, _ = _make_manifest(tmp_path)

    account_dir = accounts_dir / "7"
    account_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_type",
                    "strength": "weak",
                    "documents": ["statement"],
                    "send_to_ai": True,
                    "reason_code": "C4_TWO_MATCH_ONE_DIFF",
                    "bureau_values": {
                        "equifax": {"normalized": "conventional real estate mortgage"},
                        "experian": {"normalized": "conventional real estate mortgage"},
                        "transunion": {"normalized": "real estate mortgage"},
                    },
                },
                {
                    "field": "account_rating",
                    "strength": "soft",
                    "send_to_ai": True,
                    "reason_code": "C5_ALL_DIFF",
                    "documents": ["payment history"],
                },
            ],
            "field_consistency": {
                "account_type": {
                    "raw": {
                        "equifax": "Conventional real estate mortgage",
                        "experian": "Conventional real estate mortgage",
                        "transunion": "Real estate mortgage",
                    }
                }
            },
        }
    }

    bureaus_payload = {
        "equifax": {
            "account_type": {"raw": "Conventional real estate mortgage", "normalized": "conventional real estate mortgage"},
            "account_rating": {"raw": "1", "normalized": "1"},
        },
        "experian": {
            "account_type": {"raw": "Conventional real estate mortgage", "normalized": "conventional real estate mortgage"},
            "account_rating": {"raw": "1", "normalized": "1"},
        },
        "transunion": {
            "account_type": {"raw": "Real estate mortgage", "normalized": "real estate mortgage"},
            "account_rating": {"raw": "2", "normalized": "2"},
        },
    }

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", bureaus_payload)

    builder = ValidationPackBuilder(manifest)
    builder.build()

    pack_path = packs_dir / "val_acc_007.jsonl"
    assert pack_path.exists()

    raw_lines = [line for line in pack_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(raw_lines) == 2

    for raw_line in raw_lines:
        payload = json.loads(raw_line)
        assert payload["prompt"]["system"] == _SYSTEM_PROMPT

        finding_json = json.dumps(payload["finding"], ensure_ascii=False, sort_keys=True)
        expected_user = _PROMPT_USER_TEMPLATE.replace("<finding blob here>", finding_json)
        assert payload["prompt"]["user"] == expected_user


def test_builder_includes_style_metadata_when_available(tmp_path: Path) -> None:
    sid = "SID"
    manifest, accounts_dir, packs_dir, _ = _make_manifest(tmp_path, sid=sid)

    account_dir = accounts_dir / "8"
    account_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "account_id": "idx-008",
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_type",
                    "strength": "weak",
                    "send_to_ai": True,
                    "reason_code": "C4_TWO_MATCH_ONE_DIFF",
                }
            ],
            "field_consistency": {},
        },
    }

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", {})

    style_paths = ensure_note_style_paths(tmp_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(style_paths, "idx-008", create=True)
    account_paths.result_file.write_text(
        json.dumps(
            {
                "analysis": {
                    "tone": "confident",
                    "context_hints": {"topic": "billing_error"},
                    "emphasis": ["paid_already", "support_request"],
                },
                "prompt_salt": "salt-value-123",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    builder = ValidationPackBuilder(manifest)
    builder.build()

    pack_path = packs_dir / "val_acc_008.jsonl"
    assert pack_path.exists()

    line = next(
        json.loads(entry)
        for entry in pack_path.read_text(encoding="utf-8").splitlines()
        if entry.strip()
    )

    base_expected = _PROMPT_USER_TEMPLATE.replace(
        "<finding blob here>",
        json.dumps(line["finding"], ensure_ascii=False, sort_keys=True),
    ).rstrip()
    style_block = {
        "tone": "confident",
        "topic": "billing_error",
        "emphasis": ["paid_already", "support_request"],
        "prompt_salt": "salt-value-123",
    }
    expected_prompt = (
        f"{base_expected}\n\nSTYLE_METADATA:\n"
        f"{json.dumps(style_block, ensure_ascii=False, sort_keys=True)}"
    )

    assert line["prompt"]["user"] == expected_prompt
