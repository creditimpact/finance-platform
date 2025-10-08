import json
from pathlib import Path

import pytest

from backend.validation.build_packs import ValidationPackBuilder
from backend.validation.index_schema import load_validation_index
from backend.validation.io import read_jsonl
from backend.validation.send_packs import ValidationPackSender, _ManifestView


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


class _FailingClient:
    def create(self, *_args, **_kwargs):  # pragma: no cover - defensive safeguard
        raise AssertionError("model should not be invoked in deterministic path")


def _build_manifest(tmp_path: Path, sid: str = "SIDSEND") -> tuple[dict, Path, Path]:
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
    return manifest, accounts_dir, packs_dir


def test_sender_writes_results_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, accounts_dir, packs_dir = _build_manifest(tmp_path)

    account_dir = accounts_dir / "1"
    account_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_type",
                    "strength": "weak",
                    "send_to_ai": True,
                    "documents": ["statement"],
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
                    "documents": ["payment history"],
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

    index = load_validation_index(tmp_path / "SIDSEND" / "ai_packs" / "validation" / "index.json")
    view = _ManifestView(index=index, log_path=tmp_path / "SIDSEND" / "ai_packs" / "validation" / "validation.log")

    pack_path = index.resolve_pack_path(index.packs[0])
    pack_lines = [json.loads(line) for line in pack_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in pack_lines:
        line["send_to_ai"] = False
    pack_path.write_text(
        "\n".join(json.dumps(line, ensure_ascii=False, sort_keys=True) for line in pack_lines) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    sender = ValidationPackSender(index, http_client=_FailingClient(), preloaded_view=view)

    first_run = sender.send()
    assert len(first_run) == 1
    assert first_run[0]["status"] == "done"

    results_path = index.results_dir_path / "acc_001.result.jsonl"
    assert results_path.exists()

    result_lines = read_jsonl(results_path)
    assert len(result_lines) == 2
    expected_ids = {line["id"] for line in pack_lines}
    assert {line["id"] for line in result_lines} == expected_ids

    second_run = sender.send()
    assert len(second_run) == 1
    assert second_run[0]["status"] == "skipped"

    repeat_lines = read_jsonl(results_path)
    assert repeat_lines == result_lines
