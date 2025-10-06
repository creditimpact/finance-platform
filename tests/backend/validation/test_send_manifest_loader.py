import json
import sys
import types
from pathlib import Path

import pytest

from backend.validation.send_packs import ValidationPackSender


_requests_stub = types.ModuleType("requests")


def _post_stub(*args, **kwargs):  # pragma: no cover - safety net
    raise AssertionError("requests.post should not be invoked in tests")


_requests_stub.post = _post_stub
sys.modules.setdefault("requests", _requests_stub)


class _LoaderStubClient:
    def create(self, *, model: str, messages, response_format):  # type: ignore[override]
        payload = {
            "decision": "strong",
            "rationale": "stub",
            "citations": [],
        }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


@pytest.fixture()
def loader_client() -> _LoaderStubClient:
    return _LoaderStubClient()


def test_sender_loader_accepts_v2_manifest_mapping(
    tmp_path: Path, loader_client: _LoaderStubClient
) -> None:
    base_dir = tmp_path / "runs" / "SID555" / "ai_packs" / "validation"
    packs_dir = base_dir / "packs"
    results_dir = base_dir / "results"
    index_path = base_dir / "index.json"

    pack_file = packs_dir / "account_001.pack.jsonl"
    pack_file.parent.mkdir(parents=True, exist_ok=True)
    pack_file.write_text(
        json.dumps({"prompt": {"system": "s", "user": {"sample": True}}}) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "__index_path__": str(index_path),
        "schema_version": 2,
        "sid": "SID555",
        "root": ".",
        "packs_dir": "packs",
        "results_dir": "results",
        "packs": [
            {
                "account_id": 1,
                "pack": "packs/account_001.pack.jsonl",
                "result_jsonl": "results/account_001.result.jsonl",
                "result_json": "results/account_001.result.json",
                "lines": 1,
                "status": "built",
                "built_at": "2024-01-01T00:00:00Z",
            }
        ],
    }

    sender = ValidationPackSender(manifest, http_client=loader_client)

    index = sender._index  # type: ignore[attr-defined]
    assert index.schema_version == 2
    assert index.sid == "SID555"
    assert index.packs_dir == "packs"
    assert index.results_dir == "results"
    assert index.index_path == index_path.resolve()

    record = index.packs[0]
    assert record.pack == "packs/account_001.pack.jsonl"
    assert index.resolve_pack_path(record) == pack_file.resolve()


def test_sender_loader_converts_v1_manifest_in_memory(
    tmp_path: Path, loader_client: _LoaderStubClient
) -> None:
    base_dir = tmp_path / "runs" / "SID556" / "ai_packs" / "validation"
    packs_dir = base_dir / "packs"
    results_dir = base_dir / "results"
    index_path = base_dir / "index.json"

    pack_file = packs_dir / "account_002.pack.jsonl"
    pack_file.parent.mkdir(parents=True, exist_ok=True)
    pack_file.write_text(
        json.dumps({"prompt": {"system": "s", "user": {"sample": False}}}) + "\n",
        encoding="utf-8",
    )

    jsonl_path = results_dir / "account_002.result.jsonl"
    summary_path = results_dir / "account_002.result.json"

    manifest_v1 = {
        "__index_path__": str(index_path),
        "schema_version": 1,
        "sid": "SID556",
        "packs_dir": str(packs_dir),
        "results_dir": str(results_dir),
        "items": [
            {
                "account_id": 2,
                "pack": str(pack_file),
                "result_jsonl_path": str(jsonl_path),
                "result_path": str(summary_path),
                "lines": 1,
                "status": "built",
                "built_at": "2024-01-02T00:00:00Z",
            }
        ],
    }

    assert not index_path.exists()

    sender = ValidationPackSender(manifest_v1, http_client=loader_client)

    index = sender._index  # type: ignore[attr-defined]
    assert index.schema_version == 2
    assert index.sid == "SID556"
    assert index.packs_dir == "packs"
    assert index.results_dir == "results"
    assert index.index_path == index_path.resolve()

    record = index.packs[0]
    assert record.pack == "packs/account_002.pack.jsonl"
    assert record.result_jsonl == "results/account_002.result.jsonl"
    assert record.result_json == "results/account_002.result.json"
    assert index.resolve_pack_path(record) == pack_file.resolve()
    assert not index_path.exists(), "conversion should not persist the manifest"


def test_sender_uses_validation_stage_paths_from_run_manifest(
    tmp_path: Path, loader_client: _LoaderStubClient
) -> None:
    sid = "SID557"
    base_dir = tmp_path / "runs" / sid / "ai_packs" / "validation"
    packs_dir = base_dir / "packs"
    results_dir = base_dir / "results"
    log_path = base_dir / "logs.txt"
    index_path = base_dir / "index.json"

    packs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    index_payload = {
        "schema_version": 2,
        "sid": sid,
        "root": ".",
        "packs_dir": "packs",
        "results_dir": "results",
        "packs": [],
    }
    index_path.write_text(json.dumps(index_payload), encoding="utf-8")

    run_manifest = {
        "sid": sid,
        "ai": {
            "packs": {
                "validation": {
                    "base": str(base_dir),
                    "packs_dir": str(packs_dir),
                    "results_dir": str(results_dir),
                    "index": str(index_path),
                    "logs": str(log_path),
                }
            }
        },
        "__index_path__": str(tmp_path / "wrong" / "index.json"),
    }

    sender = ValidationPackSender(run_manifest, http_client=loader_client)

    index = sender._index  # type: ignore[attr-defined]
    assert index.index_path == index_path.resolve()
    assert index.packs_dir_path == packs_dir.resolve()
    assert index.results_dir_path == results_dir.resolve()
    assert sender._log_path == log_path.resolve()  # type: ignore[attr-defined]


def test_sender_falls_back_to_ai_validation_dir(
    tmp_path: Path, loader_client: _LoaderStubClient
) -> None:
    sid = "SID558"
    base_dir = tmp_path / "runs" / sid / "ai" / "validation"
    packs_dir = base_dir / "packs"
    results_dir = base_dir / "results"
    index_path = base_dir / "index.json"
    log_path = base_dir / "logs.txt"

    packs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    index_payload = {
        "schema_version": 2,
        "sid": sid,
        "root": ".",
        "packs_dir": "packs",
        "results_dir": "results",
        "packs": [],
    }
    index_path.write_text(json.dumps(index_payload), encoding="utf-8")

    manifest = {
        "sid": sid,
        "ai": {
            "validation": {
                "dir": str(base_dir),
            }
        },
    }

    sender = ValidationPackSender(manifest, http_client=loader_client)

    index = sender._index  # type: ignore[attr-defined]
    assert index.index_path == index_path.resolve()
    assert index.packs_dir_path == packs_dir.resolve()
    assert index.results_dir_path == results_dir.resolve()
    assert sender._log_path == log_path.resolve()  # type: ignore[attr-defined]
