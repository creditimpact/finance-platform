import json
from pathlib import Path

import pytest

from backend.validation.build_packs import ValidationPackBuilder
from backend.validation.index_schema import load_validation_index
from backend.validation.send_packs import ValidationPackSender, _ManifestView


class _UnusedClient:
    def create(self, *_args, **_kwargs):  # pragma: no cover - defensive safeguard
        raise AssertionError("model call is not expected during this test")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _build_manifest(tmp_path: Path, sid: str = "SIDPOLICY") -> tuple[dict, Path]:
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


def test_account_type_generic_specific_prefers_supportive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, accounts_dir = _build_manifest(tmp_path)

    account_dir = accounts_dir / "4"
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

    index = load_validation_index(tmp_path / "SIDPOLICY" / "ai_packs" / "validation" / "index.json")
    view = _ManifestView(index=index, log_path=tmp_path / "SIDPOLICY" / "ai_packs" / "validation" / "validation.log")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    sender = ValidationPackSender(index, http_client=_UnusedClient(), preloaded_view=view)

    pack_path = index.resolve_pack_path(index.packs[0])
    pack_line = json.loads(pack_path.read_text(encoding="utf-8").splitlines()[0])

    response = {
        "decision": "supportive_needs_companion",
        "rationale": "C4_TWO_MATCH_ONE_DIFF requires a companion field for escalation.",
        "citations": ["equifax: conventional real estate mortgage"],
        "checks": {
            "materiality": False,
            "supports_consumer": True,
            "doc_requirements_met": True,
            "mismatch_code": "C4_TWO_MATCH_ONE_DIFF",
        },
    }

    result, metadata = sender._build_result_line(4, 1, pack_line, response)

    assert result["decision"] in {"neutral_context_only", "supportive_needs_companion"}
    assert metadata["final_decision"] == result["decision"]
    assert result["legacy_decision"] == "no_case"
