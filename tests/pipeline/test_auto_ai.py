import json
from pathlib import Path

from backend.pipeline import auto_ai, auto_ai_tasks
from backend.pipeline.runs import RunManifest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_has_ai_merge_best_pairs_detects_candidates(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "sample"
    tags_path = runs_root / sid / "cases" / "accounts" / "11" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "ai", "with": 16}])

    assert auto_ai.has_ai_merge_best_pairs(sid, runs_root) is True


def test_has_ai_merge_best_pairs_handles_missing(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "missing"
    tags_path = runs_root / sid / "cases" / "accounts" / "22" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "human", "with": 9}])

    assert auto_ai.has_ai_merge_best_pairs(sid, runs_root) is False


def test_maybe_queue_auto_ai_pipeline_queues_when_candidates(monkeypatch, tmp_path: Path) -> None:
    sid = "queue-me"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("ENABLE_AUTO_AI_PIPELINE", "1")

    manifest = RunManifest.for_sid(sid)
    run_dir = manifest.path.parent
    accounts_dir = run_dir / "cases" / "accounts"
    tags_path = accounts_dir / "11" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "ai", "with": 16}])

    recorded: dict[str, object] = {}

    class DummyResult:
        id = "async-result"

    def fake_enqueue(*, sid: str, runs_root: str, accounts_dir: str):
        recorded["sid"] = sid
        recorded["runs_root"] = Path(runs_root)
        recorded["accounts_dir"] = Path(accounts_dir)
        return DummyResult()

    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_pipeline", fake_enqueue)

    result = auto_ai.maybe_queue_auto_ai_pipeline(
        sid,
        summary={"cases": {"dir": str(accounts_dir)}},
    )

    assert isinstance(result, DummyResult)
    assert recorded == {
        "sid": sid,
        "runs_root": runs_root,
        "accounts_dir": accounts_dir,
    }


def test_maybe_queue_auto_ai_pipeline_skips_without_candidates(monkeypatch, tmp_path: Path) -> None:
    sid = "skip-me"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("ENABLE_AUTO_AI_PIPELINE", "1")

    manifest = RunManifest.for_sid(sid)
    run_dir = manifest.path.parent
    accounts_dir = run_dir / "cases" / "accounts"
    tags_path = accounts_dir / "33" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "human", "with": 44}])

    calls: list[object] = []
    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_pipeline", lambda **_: calls.append(1))

    result = auto_ai.maybe_queue_auto_ai_pipeline(sid)

    assert result is None
    assert calls == []


def test_maybe_queue_auto_ai_pipeline_skips_when_disabled(monkeypatch, tmp_path: Path) -> None:
    sid = "disabled"
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.delenv("ENABLE_AUTO_AI_PIPELINE", raising=False)

    manifest = RunManifest.for_sid(sid)
    run_dir = manifest.path.parent
    accounts_dir = run_dir / "cases" / "accounts"
    tags_path = accounts_dir / "55" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "ai", "with": 56}])

    calls: list[object] = []
    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_pipeline", lambda **_: calls.append(1))

    result = auto_ai.maybe_queue_auto_ai_pipeline(sid)

    assert result is None
    assert calls == []
