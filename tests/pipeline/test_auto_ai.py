import json
from pathlib import Path

from backend.pipeline import auto_ai, auto_ai_tasks


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
    flag_env = {"ENABLE_AUTO_AI_PIPELINE": "1"}

    tags_path = runs_root / sid / "cases" / "accounts" / "11" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "ai", "with": 16}])

    recorded: dict[str, object] = {}

    def fake_enqueue(sid: str) -> str:
        recorded["sid"] = sid
        return "async-result"

    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_chain", fake_enqueue)

    result = auto_ai.maybe_queue_auto_ai_pipeline(
        sid,
        runs_root=runs_root,
        flag_env=flag_env,
    )

    marker_path = runs_root / sid / "ai_packs" / auto_ai.PIPELINE_MARKER_FILENAME

    assert result["queued"] is True
    assert recorded == {"sid": sid}
    assert marker_path.exists()


def test_maybe_queue_auto_ai_pipeline_skips_without_candidates(monkeypatch, tmp_path: Path) -> None:
    sid = "skip-me"
    runs_root = tmp_path / "runs"
    flag_env = {"ENABLE_AUTO_AI_PIPELINE": "1"}

    tags_path = runs_root / sid / "cases" / "accounts" / "33" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "human", "with": 44}])

    calls: list[object] = []
    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_chain", lambda sid: calls.append(sid))

    result = auto_ai.maybe_queue_auto_ai_pipeline(sid, runs_root=runs_root, flag_env=flag_env)

    assert result == {"queued": False, "reason": "no_candidates"}
    assert calls == []


def test_maybe_queue_auto_ai_pipeline_skips_when_disabled(monkeypatch, tmp_path: Path) -> None:
    sid = "disabled"
    runs_root = tmp_path / "runs"
    flag_env = {}

    tags_path = runs_root / sid / "cases" / "accounts" / "55" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "ai", "with": 56}])

    calls: list[object] = []
    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_chain", lambda sid: calls.append(sid))

    result = auto_ai.maybe_queue_auto_ai_pipeline(sid, runs_root=runs_root, flag_env=flag_env)

    assert result == {"queued": False, "reason": "disabled"}
    assert calls == []


def test_maybe_queue_auto_ai_pipeline_skips_when_marker_present(monkeypatch, tmp_path: Path) -> None:
    sid = "in-progress"
    runs_root = tmp_path / "runs"
    flag_env = {"ENABLE_AUTO_AI_PIPELINE": "1"}

    tags_path = runs_root / sid / "cases" / "accounts" / "11" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "ai", "with": 99}])

    marker_path = runs_root / sid / "ai_packs" / auto_ai.PIPELINE_MARKER_FILENAME
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text("{}", encoding="utf-8")

    calls: list[object] = []
    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_chain", lambda sid: calls.append(sid))

    result = auto_ai.maybe_queue_auto_ai_pipeline(sid, runs_root=runs_root, flag_env=flag_env)

    assert result == {"queued": False, "reason": "in_progress"}
    assert calls == []
