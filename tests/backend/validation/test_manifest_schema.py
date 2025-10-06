from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import sys
import types


_requests_stub = types.ModuleType("requests")


def _post_stub(*args, **kwargs):  # pragma: no cover - safety net
    raise AssertionError("requests.post should not be invoked in tests")


_requests_stub.post = _post_stub
sys.modules.setdefault("requests", _requests_stub)

from backend.validation.build_packs import ValidationPackBuilder
from backend.validation.manifest import check_index
from backend.validation.send_packs import ValidationPackSender
from backend.validation.index_schema import load_validation_index


@pytest.fixture(autouse=True)
def _legacy_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VALIDATION_SINGLE_RESULT_FILE", "0")


class _StubClient:
    def create(self, *, model: str, messages, response_format):  # type: ignore[override]
        payload = {
            "decision": "strong",
            "justification": "auto",
            "labels": ["semantic_review"],
            "citations": ["experian.raw"],
            "confidence": 0.82,
        }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def _build_manifest(tmp_path: Path, sid: str = "S001") -> tuple[dict[str, object], Path]:
    base = tmp_path / sid / "ai_packs" / "validation"
    packs_dir = base / "packs"
    results_dir = base / "results"
    index_path = base / "index.json"
    log_path = base / "logs.txt"
    accounts_dir = tmp_path / sid / "cases" / "accounts"
    account_dir = accounts_dir / "1"
    account_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "Account Status",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": False,
                    "documents": ["statement"],
                    "category": "identity",
                    "send_to_ai": True,
                }
            ],
            "field_consistency": {},
        }
    }
    (account_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    bureaus = {
        "transunion": {"Account Status": {"raw": "Value"}},
        "experian": {},
        "equifax": {},
    }
    (account_dir / "bureaus.json").write_text(json.dumps(bureaus), encoding="utf-8")

    manifest: dict[str, object] = {
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
    return manifest, tmp_path


def test_builder_writes_schema_v2(tmp_path: Path) -> None:
    manifest, _ = _build_manifest(tmp_path)
    builder = ValidationPackBuilder(manifest)
    records = builder.build()

    assert len(records) == 1

    index_path = tmp_path / "S001" / "ai_packs" / "validation" / "index.json"
    document = json.loads(index_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    assert document["root"] == "."
    assert document["packs_dir"] == "packs"
    assert document["results_dir"] == "results"
    entry = document["packs"][0]
    assert entry["pack"].startswith("packs/")
    assert entry["result_jsonl"].startswith("results/")
    assert entry["result_json"].startswith("results/")


def test_builder_respects_summary_findings(tmp_path: Path) -> None:
    manifest, runs_root = _build_manifest(tmp_path, sid="S010")

    account_dir = runs_root / "S010" / "cases" / "accounts" / "1"
    summary_path = account_dir / "summary.json"
    bureaus_path = account_dir / "bureaus.json"

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_type",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": False,
                },
                {
                    "field": "account_rating",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": True,
                },
            ],
            "field_consistency": {},
        }
    }
    bureaus_payload = {
        "transunion": {
            "account_type": "mortgage",
            "account_rating": "1",
        }
    }

    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
    bureaus_path.write_text(json.dumps(bureaus_payload), encoding="utf-8")

    builder = ValidationPackBuilder(manifest)
    records = builder.build()

    assert len(records) == 1
    record = records[0]
    assert record.get("weak_fields") == ["account_rating"]

    pack_files = sorted(builder.paths.packs_dir.glob("*.jsonl"))
    assert len(pack_files) == 1
    payloads = [
        json.loads(line)
        for line in pack_files[0].read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(payloads) == 1
    assert payloads[0]["field"] == "account_rating"


def test_builder_ignores_legacy_requirements(tmp_path: Path) -> None:
    manifest, runs_root = _build_manifest(tmp_path, sid="S011")

    account_dir = runs_root / "S011" / "cases" / "accounts" / "1"
    summary_path = account_dir / "summary.json"
    bureaus_path = account_dir / "bureaus.json"

    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_type",
                    "strength": "weak",
                    "is_mismatch": True,
                    "ai_needed": True,
                    "send_to_ai": False,
                }
            ],
            "requirements": [
                {
                    "field": "account_rating",
                    "strength": "weak",
                    "ai_needed": True,
                }
            ],
            "field_consistency": {},
        }
    }

    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
    bureaus_path.write_text(json.dumps({}), encoding="utf-8")

    builder = ValidationPackBuilder(manifest)
    records = builder.build()

    assert records == []

    pack_files = list(builder.paths.packs_dir.glob("*.jsonl"))
    assert pack_files == []

    log_entries = [
        json.loads(line)
        for line in builder.paths.log_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert any(entry.get("event") == "legacy_requirements_ignored" for entry in log_entries)
    assert all(entry.get("event") != "pack_created" for entry in log_entries)


def test_manifest_check_reports_missing_pack(tmp_path: Path) -> None:
    manifest, _ = _build_manifest(tmp_path, sid="S002")
    builder = ValidationPackBuilder(manifest)
    builder.build()

    index_path = tmp_path / "S002" / "ai_packs" / "validation" / "index.json"
    index = load_validation_index(index_path)

    buffer = io.StringIO()
    assert check_index(index, stream=buffer) is True

    pack_path = index.resolve_pack_path(index.packs[0])
    pack_path.unlink()

    buffer = io.StringIO()
    assert check_index(index, stream=buffer) is False
    output = buffer.getvalue()
    assert "MISSING" in output
    assert "Missing packs detected: 1 of 1." in output


def test_sender_uses_manifest_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, runs_root = _build_manifest(tmp_path, sid="S003")
    builder = ValidationPackBuilder(manifest)
    builder.build()

    monkeypatch.setenv("AI_MODEL", "stub")

    index_path = runs_root / "S003" / "ai_packs" / "validation" / "index.json"

    sender = ValidationPackSender(index_path, http_client=_StubClient())
    results = sender.send()

    assert len(results) == 1

    output = capsys.readouterr().out
    assert "MANIFEST:" in output
    assert "PACKS: 1, missing: 0" in output
    assert "[acc=001]" in output

    index = load_validation_index(index_path)
    record = index.packs[0]

    pack_path = index.resolve_pack_path(record)
    assert pack_path.is_file()

    jsonl_path = index.resolve_result_jsonl_path(record)
    summary_path = index.resolve_result_json_path(record)
    assert jsonl_path.is_file()
    assert summary_path.is_file()

    # ensure the sender wrote results to the manifest-defined location
    assert sender._results_root == index.results_dir_path


def test_sender_reports_missing_pack_with_relative_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, runs_root = _build_manifest(tmp_path, sid="S004")
    builder = ValidationPackBuilder(manifest)
    builder.build()

    index_path = runs_root / "S004" / "ai_packs" / "validation" / "index.json"
    index = load_validation_index(index_path)
    record = index.packs[0]

    pack_path = index.resolve_pack_path(record)
    pack_path.unlink()

    monkeypatch.setenv("AI_MODEL", "stub")

    sender = ValidationPackSender(index_path, http_client=_StubClient())
    results = sender.send()

    assert len(results) == 1
    payload = results[0]
    assert payload["status"] == "error"
    assert "Pack file missing: " in payload["error"]
    assert record.pack in payload["error"]

    output = capsys.readouterr().out
    assert "PACKS: 1, missing: 1" in output
    assert "[MISSING:" in output


def test_sender_includes_context_on_api_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class _ErrorClient:
        def create(self, *, model, messages, response_format):  # type: ignore[override]
            raise RuntimeError("boom")

    manifest, runs_root = _build_manifest(tmp_path, sid="S999")
    builder = ValidationPackBuilder(manifest)
    builder.build()

    monkeypatch.setenv("AI_MODEL", "stub")

    index_path = runs_root / "S999" / "ai_packs" / "validation" / "index.json"
    sender = ValidationPackSender(index_path, http_client=_ErrorClient())

    results = sender.send()

    assert len(results) == 1
    payload = results[0]
    assert payload["status"] == "error"
    assert "AI request failed for acc 001" in payload["error"]
    assert "pack=packs/" in payload["error"]
    assert "results/" in payload["error"]
    assert "AI request failed for acc 001" in payload["results"][0]["rationale"]

    output = capsys.readouterr().out
    assert "PACKS: 1, missing: 0" in output


def test_sender_supports_v1_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = tmp_path / "legacy" / "ai_packs" / "validation"
    packs_dir = base / "packs"
    results_dir = base / "results"
    packs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    pack_path = packs_dir / "account_001.jsonl"
    pack_payload = {
        "id": "line-001",
        "field": "Account Status",
        "prompt": {"system": "test", "user": {"account": 1}},
    }
    pack_path.write_text(json.dumps(pack_payload) + "\n", encoding="utf-8")

    result_jsonl_path = results_dir / "account_001.jsonl"
    result_summary_path = results_dir / "account_001.json"

    index_path = base / "index.json"
    document = {
        "schema_version": 1,
        "sid": "S005",
        "packs_dir": str(packs_dir),
        "results_dir": str(results_dir),
        "items": [
            {
                "account_id": 1,
                "pack_path": str(pack_path),
                "result_jsonl_path": str(result_jsonl_path),
                "result_path": str(result_summary_path),
                "lines": 1,
                "status": "built",
                "built_at": "2024-01-01T00:00:00Z",
            }
        ],
    }
    index_path.write_text(json.dumps(document), encoding="utf-8")

    monkeypatch.setenv("AI_MODEL", "stub")

    sender = ValidationPackSender(index_path, http_client=_StubClient())
    results = sender.send()

    assert len(results) == 1
    record = sender._index.packs[0]

    assert record.pack.startswith("packs/")
    assert record.result_jsonl.startswith("results/")
    assert record.result_json.startswith("results/")
    assert sender._index.root == "."

    summary_file = results_dir / "account_001.json"
    jsonl_file = results_dir / "account_001.jsonl"
    assert summary_file.is_file()
    assert jsonl_file.is_file()
