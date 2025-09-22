import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from backend.core.ai import adjudicator
from backend.core.io.tags import upsert_tag
from backend.pipeline import auto_ai, auto_ai_tasks
from scripts.score_bureau_pairs import ScoreComputationResult


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _issue_tag() -> dict[str, Any]:
    return {"kind": "issue", "type": "collection", "source": "scorer"}


def _merge_best_verbose(partner: int) -> dict[str, Any]:
    return {
        "kind": "merge_best",
        "decision": "ai",
        "with": partner,
        "total": 59,
        "mid": 20,
        "parts": {"balance_owed": 31},
        "aux": {"acctnum_level": "last4", "matched_fields": {"balance_owed": True}},
        "conflicts": ["credit_limit:conflict"],
        "strong": True,
    }


def _expected_minimal_tags(partner: int, *, timestamp: str) -> list[dict[str, Any]]:
    return [
        {"kind": "issue", "type": "collection"},
        {"kind": "merge_best", "decision": "ai", "with": partner},
        {
            "kind": "ai_decision",
            "decision": "merge",
            "with": partner,
            "at": timestamp,
        },
        {"kind": "same_debt_pair", "with": partner, "at": timestamp},
    ]


def test_has_ai_merge_best_pairs_guard_handles_structured_tags(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "guard"
    account_tags = runs_root / sid / "cases" / "accounts"

    _write_json(
        account_tags / "11" / "tags.json",
        {"tags": [{"kind": "merge_best", "decision": "ai", "with": 16}]},
    )
    _write_json(
        account_tags / "12" / "tags.json",
        {"tags": [{"kind": "merge_best", "decision": "human", "with": 99}]},
    )

    assert auto_ai.has_ai_merge_best_pairs(sid, runs_root) is True

    _write_json(
        account_tags / "11" / "tags.json",
        {"tags": [{"kind": "merge_best", "decision": "human", "with": 16}]},
    )

    assert auto_ai.has_ai_merge_best_pairs(sid, runs_root) is False


def test_has_ai_merge_best_pairs_detects_missing_partner(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "no-partner"
    account_tags = runs_root / sid / "cases" / "accounts"

    _write_json(account_tags / "11" / "tags.json", [{"kind": "merge_best", "decision": "ai"}])

    assert auto_ai.has_ai_merge_best_pairs(sid, runs_root) is True


def test_maybe_queue_auto_ai_pipeline_queues_when_candidates(monkeypatch, tmp_path: Path) -> None:
    sid = "queue-me"
    runs_root = tmp_path / "runs"
    flag_env = {"ENABLE_AUTO_AI_PIPELINE": "1"}

    tags_path = runs_root / sid / "cases" / "accounts" / "11" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "ai", "with": 16}])

    recorded: dict[str, Any] = {}

    def fake_enqueue(sid_value: str, runs_root=None) -> str:
        recorded["sid"] = sid_value
        recorded["runs_root"] = runs_root
        return "async-result"

    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_chain", fake_enqueue)

    result = auto_ai.maybe_queue_auto_ai_pipeline(
        sid,
        runs_root=runs_root,
        flag_env=flag_env,
    )

    lock_path = (
        runs_root
        / sid
        / auto_ai.AUTO_AI_PIPELINE_DIRNAME
        / auto_ai.INFLIGHT_LOCK_FILENAME
    )

    assert result["queued"] is True
    assert result["reason"] == "queued"
    assert recorded == {"sid": sid, "runs_root": runs_root}
    assert lock_path.exists()
    assert result["lock_path"] == str(lock_path)
    assert result["pipeline_dir"] == str(lock_path.parent)
    assert result["last_ok_path"] == str(
        lock_path.parent / auto_ai.LAST_OK_FILENAME
    )


def test_maybe_queue_auto_ai_pipeline_skips_without_candidates(monkeypatch, tmp_path: Path) -> None:
    sid = "skip-me"
    runs_root = tmp_path / "runs"
    flag_env = {"ENABLE_AUTO_AI_PIPELINE": "1"}

    tags_path = runs_root / sid / "cases" / "accounts" / "33" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "human", "with": 44}])

    calls: list[object] = []
    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_chain", lambda sid_value, runs_root=None: calls.append((sid_value, runs_root)))

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
    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_chain", lambda sid_value, runs_root=None: calls.append((sid_value, runs_root)))

    result = auto_ai.maybe_queue_auto_ai_pipeline(sid, runs_root=runs_root, flag_env=flag_env)

    assert result == {"queued": False, "reason": "disabled"}
    assert calls == []


def test_maybe_queue_auto_ai_pipeline_skips_when_lock_present(monkeypatch, tmp_path: Path) -> None:
    sid = "in-progress"
    runs_root = tmp_path / "runs"
    flag_env = {"ENABLE_AUTO_AI_PIPELINE": "1"}

    tags_path = runs_root / sid / "cases" / "accounts" / "11" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "ai", "with": 99}])

    lock_path = (
        runs_root
        / sid
        / auto_ai.AUTO_AI_PIPELINE_DIRNAME
        / auto_ai.INFLIGHT_LOCK_FILENAME
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{}", encoding="utf-8")

    calls: list[object] = []
    monkeypatch.setattr(
        auto_ai_tasks,
        "enqueue_auto_ai_chain",
        lambda sid_value, runs_root=None: calls.append((sid_value, runs_root)),
    )

    result = auto_ai.maybe_queue_auto_ai_pipeline(sid, runs_root=runs_root, flag_env=flag_env)

    assert result == {"queued": False, "reason": "inflight"}
    assert calls == []


def test_maybe_queue_auto_ai_pipeline_clears_stale_lock(monkeypatch, tmp_path: Path) -> None:
    sid = "stale"
    runs_root = tmp_path / "runs"
    flag_env = {"ENABLE_AUTO_AI_PIPELINE": "1"}

    tags_path = runs_root / sid / "cases" / "accounts" / "11" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "ai", "with": 42}])

    lock_path = (
        runs_root
        / sid
        / auto_ai.AUTO_AI_PIPELINE_DIRNAME
        / auto_ai.INFLIGHT_LOCK_FILENAME
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{}", encoding="utf-8")
    old_age = auto_ai.DEFAULT_INFLIGHT_TTL_SECONDS + 5
    past = time.time() - old_age
    os.utime(lock_path, (past, past))

    recorded: dict[str, Any] = {}

    def fake_enqueue(sid_value: str, runs_root=None) -> str:
        recorded["sid"] = sid_value
        recorded["runs_root"] = runs_root
        return "queued"

    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_chain", fake_enqueue)

    result = auto_ai.maybe_queue_auto_ai_pipeline(
        sid,
        runs_root=runs_root,
        flag_env=flag_env,
    )

    assert result["queued"] is True
    assert recorded == {"sid": sid, "runs_root": runs_root}


def test_maybe_queue_auto_ai_pipeline_force_overrides_lock(monkeypatch, tmp_path: Path) -> None:
    sid = "force"
    runs_root = tmp_path / "runs"
    flag_env = {"ENABLE_AUTO_AI_PIPELINE": "1"}

    tags_path = runs_root / sid / "cases" / "accounts" / "11" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "ai", "with": 11}])

    lock_path = (
        runs_root
        / sid
        / auto_ai.AUTO_AI_PIPELINE_DIRNAME
        / auto_ai.INFLIGHT_LOCK_FILENAME
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("{}", encoding="utf-8")

    recorded: dict[str, Any] = {}

    def fake_enqueue(sid_value: str, runs_root=None) -> str:
        recorded["sid"] = sid_value
        recorded["runs_root"] = runs_root
        return "queued"

    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_chain", fake_enqueue)

    result = auto_ai.maybe_queue_auto_ai_pipeline(
        sid,
        runs_root=runs_root,
        flag_env=flag_env,
        force=True,
    )

    assert result["queued"] is True
    assert recorded == {"sid": sid, "runs_root": runs_root}


def test_enqueue_auto_ai_chain_orders_signatures(monkeypatch) -> None:
    sid = "sig-order"
    runs_root = Path("/tmp/runs")

    recorded_signatures: list[tuple[str, tuple[Any, ...]]] = []

    class _Recorder:
        def __init__(self, name: str) -> None:
            self.name = name
            self.calls: list[tuple[str, tuple[Any, ...]]] = []

        def s(self, *args: Any) -> tuple[str, tuple[Any, ...]]:
            signature = (self.name, args)
            self.calls.append(signature)
            recorded_signatures.append(signature)
            return signature

    score_recorder = _Recorder("score")
    build_recorder = _Recorder("build")
    send_recorder = _Recorder("send")
    compact_recorder = _Recorder("compact")

    monkeypatch.setattr(auto_ai_tasks, "ai_score_step", score_recorder)
    monkeypatch.setattr(auto_ai_tasks, "ai_build_packs_step", build_recorder)
    monkeypatch.setattr(auto_ai_tasks, "ai_send_packs_step", send_recorder)
    monkeypatch.setattr(auto_ai_tasks, "ai_compact_tags_step", compact_recorder)

    chain_calls: dict[str, Any] = {}

    class _FakeWorkflow:
        def __init__(self, steps: tuple[tuple[str, tuple[Any, ...]], ...]) -> None:
            self.steps = steps

        def apply_async(self) -> "_FakeResult":
            chain_calls["apply_async"] = self.steps
            return _FakeResult()

    class _FakeResult:
        id = "chain-root-task"

    def fake_chain(*steps: tuple[str, tuple[Any, ...]]) -> _FakeWorkflow:
        chain_calls["steps"] = steps
        return _FakeWorkflow(steps)

    monkeypatch.setattr(auto_ai_tasks, "chain", fake_chain)

    task_id = auto_ai_tasks.enqueue_auto_ai_chain(sid, runs_root=runs_root)

    assert task_id == "chain-root-task"
    assert chain_calls["steps"] == (
        ("score", (sid, str(runs_root))),
        ("build", ()),
        ("send", ()),
        ("compact", ()),
    )
    assert chain_calls["apply_async"] == chain_calls["steps"]


def test_auto_ai_chain_idempotent_and_compacts_tags(monkeypatch, tmp_path: Path) -> None:
    sid = "auto-chain"
    runs_root = tmp_path / "runs"
    timestamp = "2024-07-04T12:00:00Z"

    account_root = runs_root / sid / "cases" / "accounts"
    account_a = account_root / "11"
    account_b = account_root / "16"

    _write_json(account_a / "tags.json", [_issue_tag(), _merge_best_verbose(16)])
    _write_json(account_b / "tags.json", [_issue_tag(), _merge_best_verbose(11)])

    expected_runs_root = runs_root

    def fake_score_accounts(
        sid_value: str,
        *,
        runs_root: Path | str,
        only_ai_rows: bool = False,
        write_tags: bool = False,
    ) -> ScoreComputationResult:
        assert sid_value == sid
        runs_root_path = Path(runs_root)
        assert runs_root_path == expected_runs_root
        assert write_tags is True
        return ScoreComputationResult(
            sid=sid_value,
            runs_root=runs_root_path,
            indices=[11, 16],
            scores_by_idx={},
            best_by_idx={},
            merge_tags={},
            rows=[],
        )

    def fake_build_packs(sid_value: str, runs_root_value: Path) -> None:
        assert sid_value == sid
        packs_dir = runs_root_value / sid_value / "ai_packs"
        packs_dir.mkdir(parents=True, exist_ok=True)
        pack_payload = {"sid": sid_value, "pair": {"a": 11, "b": 16}, "context": []}
        _write_json(packs_dir / "011-016.json", pack_payload)
        _write_json(packs_dir / "index.json", [{"a": 11, "b": 16, "file": "011-016.json"}])

    def fake_send_packs(sid_value: str, runs_root: Path | None = None) -> None:
        assert sid_value == sid
        base_root = Path(runs_root) if runs_root is not None else Path("runs")
        run_dir = base_root / sid_value
        for source_idx, partner_idx in ((11, 16), (16, 11)):
            decision_tag = {
                "kind": "ai_decision",
                "tag": "ai_decision",
                "source": "ai_adjudicator",
                "with": partner_idx,
                "decision": "merge",
                "reason": "Records align cleanly.",
                "at": timestamp,
            }
            same_debt_tag = {
                "kind": "same_debt_pair",
                "with": partner_idx,
                "source": "ai_adjudicator",
                "reason": "Records align cleanly.",
                "at": timestamp,
            }
            account_dir = run_dir / "cases" / "accounts" / f"{source_idx}"
            upsert_tag(account_dir, decision_tag, unique_keys=("kind", "with", "source"))
            upsert_tag(account_dir, same_debt_tag, unique_keys=("kind", "with", "source"))

    monkeypatch.setattr(auto_ai_tasks, "score_accounts", fake_score_accounts)
    monkeypatch.setattr(auto_ai_tasks, "_build_ai_packs", fake_build_packs)
    monkeypatch.setattr(auto_ai_tasks, "_send_ai_packs", fake_send_packs)

    payload = auto_ai_tasks.ai_score_step.run(sid, str(runs_root))
    payload = auto_ai_tasks.ai_build_packs_step.run(payload)
    payload = auto_ai_tasks.ai_send_packs_step.run(payload)
    first_result = auto_ai_tasks.ai_compact_tags_step.run(payload)

    packs_index = json.loads(
        (runs_root / sid / "ai_packs" / "index.json").read_text(encoding="utf-8")
    )
    assert packs_index == [{"a": 11, "b": 16, "file": "011-016.json"}]
    assert first_result["packs"] == 1
    assert first_result["pairs"] == 2

    logs_path = runs_root / sid / "ai_packs" / "logs.txt"
    first_logs = [
        json.loads(line)
        for line in logs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(first_logs) == 1
    assert first_logs[0]["packs"] == 1
    assert first_logs[0]["pairs"] == 2

    tags_a_first = json.loads((account_a / "tags.json").read_text(encoding="utf-8"))
    tags_b_first = json.loads((account_b / "tags.json").read_text(encoding="utf-8"))
    summary_a_first = json.loads((account_a / "summary.json").read_text(encoding="utf-8"))
    summary_b_first = json.loads((account_b / "summary.json").read_text(encoding="utf-8"))

    assert tags_a_first == _expected_minimal_tags(16, timestamp=timestamp)
    assert tags_b_first == _expected_minimal_tags(11, timestamp=timestamp)
    assert summary_a_first["merge_explanations"][0]["with"] == 16
    assert summary_b_first["ai_explanations"][0]["decision"] == "merge"

    payload = auto_ai_tasks.ai_score_step.run(sid, str(runs_root))
    payload = auto_ai_tasks.ai_build_packs_step.run(payload)
    payload = auto_ai_tasks.ai_send_packs_step.run(payload)
    second_result = auto_ai_tasks.ai_compact_tags_step.run(payload)

    assert second_result["packs"] == 1
    assert second_result["pairs"] == 2

    second_logs = [
        json.loads(line)
        for line in logs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [entry["packs"] for entry in second_logs] == [1, 1]
    assert [entry["pairs"] for entry in second_logs] == [2, 2]

    tags_a_second = json.loads((account_a / "tags.json").read_text(encoding="utf-8"))
    tags_b_second = json.loads((account_b / "tags.json").read_text(encoding="utf-8"))
    summary_a_second = json.loads((account_a / "summary.json").read_text(encoding="utf-8"))
    summary_b_second = json.loads((account_b / "summary.json").read_text(encoding="utf-8"))

    assert tags_a_second == tags_a_first
    assert tags_b_second == tags_b_first
    assert summary_a_second == summary_a_first
    assert summary_b_second == summary_b_first


class _FakeResponse:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        if payload is None:
            payload = {"decision": "merge", "reason": "test"}
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(self._payload, ensure_ascii=False)
                    }
                }
            ]
        }


def test_auto_ai_project_key_header(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OPENAI_API_KEY", "AI_MODEL", "OPENAI_PROJECT_ID"):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abc123")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-789")
    monkeypatch.setenv("AI_MODEL", "merge-model")

    recorded: dict[str, Any] = {}

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: int) -> _FakeResponse:
        recorded["url"] = url
        recorded["headers"] = dict(headers)
        recorded["json"] = json
        recorded["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(adjudicator.httpx, "post", fake_post)

    result = adjudicator.decide_merge_or_different({}, timeout=7)

    assert result == {"decision": "merge", "reason": "test"}
    assert recorded["headers"]["Authorization"].startswith("Bearer sk-proj-abc123")
    assert recorded["headers"]["OpenAI-Project"] == "proj-789"
