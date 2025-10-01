import json
import sys
import types
from pathlib import Path


_requests_stub = types.ModuleType("requests")


def _post_stub(*args, **kwargs):  # pragma: no cover - safety net
    raise AssertionError("requests.post should not be invoked in tests")


_requests_stub.post = _post_stub
sys.modules.setdefault("requests", _requests_stub)


from backend.ai.validation_index import ValidationIndexEntry, write_validation_manifest_v2


def test_write_validation_manifest_v2_uses_relative_posix_paths(tmp_path: Path) -> None:
    base_dir = tmp_path / "runs" / "SID123" / "ai_packs" / "validation"
    packs_dir = base_dir / "packs"
    results_dir = base_dir / "results"
    index_path = base_dir / "index.json"

    pack_path = packs_dir / "nested" / "pack.jsonl"
    result_jsonl_path = results_dir / "nested" / "account_001.result.jsonl"
    result_json_path = results_dir / "nested" / "account_001.result.json"

    entry = ValidationIndexEntry(
        account_id=1,
        pack_path=pack_path,
        result_jsonl_path=result_jsonl_path,
        result_json_path=result_json_path,
        weak_fields=("Account Name",),
        line_count=7,
        status="built",
        built_at="2024-01-01T00:00:00Z",
        extra={"custom": "value"},
    )

    write_validation_manifest_v2(
        "SID123",
        packs_dir,
        results_dir,
        [entry],
        index_path=index_path,
    )

    document = json.loads(index_path.read_text(encoding="utf-8"))

    assert document["schema_version"] == 2
    assert document["sid"] == "SID123"
    assert document["root"] == "."
    assert document["packs_dir"] == "packs"
    assert document["results_dir"] == "results"

    assert len(document["packs"]) == 1
    record = document["packs"][0]

    assert record["account_id"] == 1
    assert record["pack"] == "packs/nested/pack.jsonl"
    assert record["result_jsonl"] == "results/nested/account_001.result.jsonl"
    assert record["result_json"] == "results/nested/account_001.result.json"
    assert "\\" not in record["pack"]
    assert "\\" not in record["result_jsonl"]
    assert "\\" not in record["result_json"]

    assert record["weak_fields"] == ["Account Name"]
    assert record["lines"] == 7
    assert record["status"] == "built"
    assert record["built_at"] == "2024-01-01T00:00:00Z"
    assert record["custom"] == "value"

    expected_keys = {
        "account_id",
        "pack",
        "result_jsonl",
        "result_json",
        "weak_fields",
        "lines",
        "status",
        "built_at",
        "custom",
    }
    assert set(record) == expected_keys
