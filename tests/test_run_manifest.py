import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

try:  # pragma: no cover - optional dependency shim for test envs
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    requests = ModuleType("requests")
    requests.Session = object  # type: ignore[attr-defined]
    sys.modules["requests"] = requests

from backend.core.paths.frontend_review import ensure_frontend_review_dirs
from backend.pipeline.runs import (
    RunManifest,
    RUNS_ROOT_ENV,
    MANIFEST_ENV,
    write_breadcrumb,
)
from backend.runflow.manifest import update_manifest_frontend


def test_run_manifest_basic(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))
    m = RunManifest.for_sid("sid123")

    # index and current files
    idx = runs_root / "index.json"
    cur = runs_root / "current.txt"
    assert idx.exists()
    assert cur.read_text() == "sid123"

    # snapshot env
    monkeypatch.setenv("FOO", "bar")
    m.snapshot_env(["FOO", "MISSING"])
    assert m.data["env_snapshot"] == {"FOO": "bar"}

    # base dir and artifacts are stored as absolute paths
    base_dir = tmp_path / "traces" / "blocks" / m.sid / "accounts_table"
    m.set_base_dir("traces_accounts_table", base_dir)
    sample_file = base_dir / "accounts.json"
    base_dir.mkdir(parents=True)
    sample_file.write_text("{}")
    m.set_artifact("traces.accounts_table", "accounts_json", sample_file)

    resolved = str(sample_file.resolve())
    assert m.get("traces.accounts_table", "accounts_json") == resolved
    assert m.data["base_dirs"]["traces_accounts_table"] == str(base_dir.resolve())


def test_run_manifest_ensure_run_subdir(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))
    m = RunManifest.for_sid("sid123")

    traces_dir = m.ensure_run_subdir("traces_dir", "traces")

    expected = (runs_root / "sid123" / "traces").resolve()
    assert traces_dir == expected
    assert traces_dir.exists()
    assert m.data["base_dirs"]["traces_dir"] == str(expected)


def test_run_manifest_from_env_or_latest(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    m1 = RunManifest.for_sid("a")
    time.sleep(1)
    m2 = RunManifest.for_sid("b")

    monkeypatch.setenv(MANIFEST_ENV, str(m1.path))
    chosen = RunManifest.from_env_or_latest()
    assert chosen.sid == "a"

    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    chosen = RunManifest.from_env_or_latest()
    assert chosen.sid == "b"


def test_write_breadcrumb(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))
    m = RunManifest.for_sid("xyz")

    trace_dir = tmp_path / "traces" / "blocks" / m.sid / "accounts_table"
    trace_dir.mkdir(parents=True)
    breadcrumb = trace_dir / ".manifest"
    write_breadcrumb(m.path, breadcrumb)
    assert breadcrumb.read_text() == str(m.path.resolve())


def test_frontend_review_structure(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))

    manifest = RunManifest.for_sid("sid-review")
    run_dir = runs_root / "sid-review"

    canonical = ensure_frontend_review_dirs(str(run_dir))
    packs_dir = Path(canonical["packs_dir"])
    responses_dir = Path(canonical["responses_dir"])

    (packs_dir / "idx-0001.json").write_text("{}")
    (packs_dir / "idx-0002.json").write_text("{}")

    updated_manifest = update_manifest_frontend(
        "sid-review",
        packs_dir=packs_dir,
        packs_count=0,
        built=True,
        last_built_at=None,
        manifest=manifest,
    )

    assert packs_dir.exists()
    pack_files = list(packs_dir.glob("*.json"))
    assert len(pack_files) == 2

    assert responses_dir.exists()

    frontend_block = updated_manifest.data["frontend"]
    for key in ("dir", "packs", "results", "index"):
        assert frontend_block[key]
    assert frontend_block["packs_count"] > 0

    assert not (run_dir / "frontend" / "accounts").exists()
