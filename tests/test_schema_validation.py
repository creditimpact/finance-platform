import json
from pathlib import Path

import pytest

from backend.validation.build_packs import ValidationPackBuilder
from backend.validation.index_schema import load_validation_index
from backend.validation.io import read_jsonl
from backend.validation.send_packs import ValidationPackSender, _ManifestView


class _InvalidJsonClient:
    def create(self, *_, pack_id=None, on_error=None, **__):  # pragma: no cover - sanity guard
        if on_error is not None:
            try:
                on_error(200, "{not json")
            except Exception:
                pass
        return {
            "choices": [{"message": {"content": "{not json"}}],
            "status_code": 200,
            "latency": 0.01,
            "retries": 0,
        }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _build_manifest(tmp_path: Path, sid: str = "SIDSCHEMA") -> tuple[dict, Path]:
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
    return manifest, accounts_dir


def test_schema_fallback_on_invalid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, accounts_dir = _build_manifest(tmp_path)

    account_dir = accounts_dir / "11"
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
                }
            ],
            "field_consistency": {},
        }
    }

    _write_json(account_dir / "summary.json", summary_payload)
    _write_json(account_dir / "bureaus.json", {})

    builder = ValidationPackBuilder(manifest)
    builder.build()

    index = load_validation_index(tmp_path / "SIDSCHEMA" / "ai_packs" / "validation" / "index.json")
    view = _ManifestView(index=index, log_path=tmp_path / "SIDSCHEMA" / "ai_packs" / "validation" / "validation.log")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    sender = ValidationPackSender(index, http_client=_InvalidJsonClient(), preloaded_view=view)

    summaries = sender.send()
    assert len(summaries) == 1
    assert summaries[0]["status"] == "done"

    results_path = index.results_dir_path / "acc_011.result.jsonl"
    result_lines = read_jsonl(results_path)
    assert len(result_lines) == 1
    fallback = result_lines[0]
    assert fallback["decision"] == "no_case"
    assert "schema_mismatch" in fallback["rationale"]
    assert fallback["citations"] == ["system:none"]
