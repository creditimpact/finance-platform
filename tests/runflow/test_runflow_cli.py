import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

if "requests" not in sys.modules:
    module = ModuleType("requests")

    class _DummySession:
        def get(self, *_args, **_kwargs):
            return SimpleNamespace(status_code=200, headers={}, text="")

        def close(self) -> None:
            pass

    module.Session = _DummySession
    module.RequestException = Exception
    sys.modules["requests"] = module

from scripts import runflow as runflow_cli
from backend.validation.index_schema import (
    ValidationIndex,
    ValidationPackRecord,
    load_validation_index,
)


def _write_validation_index(run_dir: Path, sid: str, *, status: str) -> None:
    validation_dir = run_dir / "ai_packs" / "validation"
    packs_dir = validation_dir / "packs"
    results_dir = validation_dir / "results"
    packs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    pack_path = packs_dir / "idx-001.json"
    pack_path.write_text("{}", encoding="utf-8")

    result_path = results_dir / "idx-001.result.jsonl"
    result_path.write_text("{}", encoding="utf-8")

    record = ValidationPackRecord(
        account_id=1,
        pack="packs/idx-001.json",
        result_json=f"results/{result_path.name}",
        result_jsonl=None,
        lines=1,
        status=status,
        built_at="2024-01-01T00:00:00Z",
    )
    index = ValidationIndex(
        index_path=validation_dir / "index.json",
        sid=sid,
        packs_dir="packs",
        results_dir="results",
        packs=[record],
    )
    index.write()


def test_backfill_validation_updates_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    runs_root = tmp_path / "runs"
    sid = "SID-CLI"
    run_dir = runs_root / sid

    _write_validation_index(run_dir, sid, status="built")

    exit_code = runflow_cli.main(
        ["backfill-validation", sid, "--runs-root", str(runs_root)]
    )

    assert exit_code == 0

    captured = json.loads(capsys.readouterr().out)
    assert captured["sid"] == sid
    assert captured["updated"] == 1

    loaded_index = load_validation_index(run_dir / "ai_packs" / "validation" / "index.json")
    assert loaded_index.packs[0].status == "completed"

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    validation_stage = runflow_payload["stages"]["validation"]
    assert validation_stage["status"] == "success"
    umbrella = runflow_payload["umbrella_barriers"]
    assert umbrella["validation_ready"] is True
