import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pytest

from backend.core.ai import adjudicator
from backend.core.io.tags import upsert_tag
from backend.pipeline import auto_ai, auto_ai_tasks
from backend.pipeline.runs import RunManifest
from scripts.score_bureau_pairs import ScoreComputationResult


from tests.scripts.test_send_ai_merge_packs import (
    _merge_best_tag,
    _write_raw_lines,
)

from scripts.build_ai_merge_packs import main as build_ai_merge_packs_main
from scripts.send_ai_merge_packs import main as send_ai_merge_packs_main
from backend.core.logic.report_analysis.tags_compact import compact_tags_for_account



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
        "total": 53,
        "mid": 20,
        "parts": {"balance_owed": 31, "account_number": 22},
        "aux": {
            "acctnum_level": "last4",
            "matched_fields": {"balance_owed": True, "account_number": True},
        },
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
            "flags": {"account_match": True, "debt_match": True},
        },
        {"kind": "same_account_pair", "with": partner, "at": timestamp},
        {
            "kind": "ai_resolution",
            "with": partner,
            "decision": "merge",
            "flags": {"account_match": True, "debt_match": True},
            "reason": "Records align cleanly.",
        },
    ]




def _setup_merge_case(runs_root: Path, sid: str = "codex-flow") -> tuple[str, Path, Path]:
    accounts_root = runs_root / sid / "cases" / "accounts"
    account_a = accounts_root / "11"
    account_b = accounts_root / "16"
    _write_raw_lines(
        account_a / "raw_lines.json",
        [
            "US BK CACS",
            "Account # 409451******",
            "Balance Owed: $12,091",
            "Last Payment: 2024-02-11",
            "Creditor Remarks: Transferred",
        ],
    )
    _write_raw_lines(
        account_b / "raw_lines.json",
        [
            "U S BANK",
            "Account # -- 409451******",
            "Balance Owed: -- $12,091 --",
            "Last Payment: 2024-02-11",
            "Remarks: Referred to collections",
        ],
    )
    _write_json(account_a / "tags.json", [_merge_best_tag(16)])
    _write_json(account_b / "tags.json", [_merge_best_tag(11)])
    return sid, account_a, account_b

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


def test_has_ai_merge_best_pairs_skips_zero_debt(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    runs_root = tmp_path / "runs"
    sid = "zero-guard"
    accounts = runs_root / sid / "cases" / "accounts"

    _write_json(accounts / "11" / "tags.json", [{"kind": "merge_best", "decision": "ai", "with": 16}])
    _write_json(accounts / "16" / "tags.json", [{"kind": "merge_best", "decision": "ai", "with": 11}])

    zero_payload = {"balance_owed": 0, "past_due_amount": 0}
    _write_json(accounts / "11" / "fields_flat.json", zero_payload)
    _write_json(accounts / "16" / "fields_flat.json", zero_payload)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="backend.pipeline.auto_ai"):
        assert auto_ai.has_ai_merge_best_pairs(sid, runs_root) is False

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        f"AI_CANDIDATE_SKIPPED_ZERO_DEBT sid={sid} a=11 b=16" in message
        for message in messages
    )


def test_maybe_queue_auto_ai_pipeline_queues_when_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sid = "queue-me"
    runs_root = tmp_path / "runs"
    flag_env = {"ENABLE_AUTO_AI_PIPELINE": "1"}

    tags_path = runs_root / sid / "cases" / "accounts" / "11" / "tags.json"
    _write_json(tags_path, [{"kind": "merge_best", "decision": "ai", "with": 16}])

    recorded: dict[str, Any] = {}

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    def fake_enqueue(sid_value: str, runs_root=None) -> str:
        recorded["sid"] = sid_value
        recorded["runs_root"] = runs_root
        return "async-result"

    monkeypatch.setattr(auto_ai_tasks, "enqueue_auto_ai_chain", fake_enqueue)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="backend.pipeline.auto_ai"):
        result = auto_ai.maybe_queue_auto_ai_pipeline(
            sid,
            runs_root=runs_root,
            flag_env=flag_env,
        )
    messages = [record.getMessage() for record in caplog.records]

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

    assert any(f"MANIFEST_AI_ENQUEUED sid={sid}" in message for message in messages)

    manifest = RunManifest.for_sid(sid)
    status = manifest.data.get("ai", {}).get("status", {})
    assert status.get("enqueued") is True
    assert status.get("built") is False
    assert status.get("sent") is False
    assert status.get("compacted") is False
    assert status.get("skipped_reason") is None


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
        pack_payload = {
            "sid": sid_value,
            "pair": {"a": 11, "b": 16},
            "context": {"a": [], "b": []},
            "highlights": {"total": 0},
        }
        pack_filename = "pair_011_016.jsonl"
        (packs_dir / pack_filename).write_text(json.dumps(pack_payload, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_json(
            packs_dir / "index.json",
            {
                "sid": sid_value,
                "pairs": [
                    {
                        "a": 11,
                        "b": 16,
                        "pack_file": pack_filename,
                        "lines_a": 0,
                        "lines_b": 0,
                        "score_total": 0,
                    }
                ],
            },
        )

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
                "flags": {"account_match": True, "debt_match": True},
                "at": timestamp,
            }
            pair_tag = {
                "kind": "same_account_pair",
                "with": partner_idx,
                "source": "ai_adjudicator",
                "reason": "Records align cleanly.",
                "at": timestamp,
            }
            account_dir = run_dir / "cases" / "accounts" / f"{source_idx}"
            upsert_tag(account_dir, decision_tag, unique_keys=("kind", "with", "source"))
            upsert_tag(account_dir, pair_tag, unique_keys=("kind", "with", "source"))

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
    assert packs_index == {"sid": sid, "pairs": [{"a": 11, "b": 16, "pack_file": "pair_011_016.jsonl", "lines_a": 0, "lines_b": 0, "score_total": 0}]}
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
    assert any(entry.get("origin") == "ai" for entry in summary_a_first["merge_explanations"])
    assert summary_b_first["ai_explanations"][0]["decision"] == "merge"
    assert summary_b_first["ai_explanations"][0]["normalized"] is False

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


def test_maybe_run_ai_pipeline_skips_without_candidates_logs(monkeypatch, tmp_path, caplog):
    runs_root = tmp_path / "runs"
    sid = "no-ai"
    account_dir = runs_root / sid / "cases" / "accounts" / "01"
    _write_json(account_dir / "tags.json", [{"kind": "merge_best", "decision": "human", "with": 7}])

    monkeypatch.setenv("ENABLE_AUTO_AI_PIPELINE", "1")
    monkeypatch.setattr(auto_ai, "RUNS_ROOT", runs_root)

    def _boom(_sid: str):  # pragma: no cover - defensive
        raise AssertionError("pipeline should not run when no candidates")

    monkeypatch.setattr(auto_ai, "_run_auto_ai_pipeline", _boom)

    with caplog.at_level("INFO"):
        result = auto_ai.maybe_run_ai_pipeline(sid)

    assert result == {"sid": sid, "skipped": "no_ai_candidates"}
    manifest = RunManifest.for_sid(sid)
    status = manifest.data.get("ai", {}).get("status", {})
    assert status.get("skipped_reason") == "no_candidates"
    assert status.get("built") is False
    assert status.get("sent") is False
    assert status.get("compacted") is False
    assert any("AUTO_AI_SKIPPED" in record.getMessage() for record in caplog.records)


def test_maybe_run_ai_pipeline_skips_zero_debt_candidates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runs_root = tmp_path / "runs"
    sid = "zero-skip"
    account_root = runs_root / sid / "cases" / "accounts"

    _write_json(account_root / "11" / "tags.json", [{"kind": "merge_best", "decision": "ai", "with": 16}])
    _write_json(account_root / "16" / "tags.json", [{"kind": "merge_best", "decision": "ai", "with": 11}])

    zero_payload = {"balance_owed": 0, "past_due_amount": 0}
    _write_json(account_root / "11" / "fields_flat.json", zero_payload)
    _write_json(account_root / "16" / "fields_flat.json", zero_payload)

    monkeypatch.setenv("ENABLE_AUTO_AI_PIPELINE", "1")
    monkeypatch.setattr(auto_ai, "RUNS_ROOT", runs_root)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="backend.pipeline.auto_ai"):
        result = auto_ai.maybe_run_ai_pipeline(sid)

    assert result == {"sid": sid, "skipped": "no_ai_candidates"}

    manifest = RunManifest.for_sid(sid)
    status = manifest.data.get("ai", {}).get("status", {})
    assert status.get("skipped_reason") == "no_candidates"

    messages = [record.getMessage() for record in caplog.records]
    assert any(f"AUTO_AI_SKIPPED sid={sid} reason=no_candidates" in message for message in messages)
    assert any(
        f"AI_CANDIDATE_SKIPPED_ZERO_DEBT sid={sid} a=11 b=16" in message
        for message in messages
    )


def test_auto_ai_build_and_send_use_ai_packs_dir(tmp_path, monkeypatch, caplog):
    runs_root = tmp_path / "runs"
    sid, account_a, account_b = _setup_merge_case(runs_root)
    packs_dir = auto_ai.packs_dir_for(sid, runs_root=runs_root)

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="scripts.build_ai_merge_packs"):
        build_ai_merge_packs_main([
            "--sid",
            sid,
            "--runs-root",
            str(runs_root),
            "--max-lines-per-side",
            "6",
        ])

    assert packs_dir.exists()
    index_path = packs_dir / "index.json"
    assert index_path.exists()

    manifest = RunManifest.for_sid(sid)
    ai_info = manifest.data.get("ai", {})
    packs_info = ai_info.get("packs", {})
    status_info = ai_info.get("status", {})
    assert Path(packs_info.get("dir")) == packs_dir.resolve()
    assert Path(packs_info.get("dir")).exists()
    assert Path(packs_info.get("index")) == index_path.resolve()
    assert Path(packs_info.get("index")).exists()
    assert packs_info.get("pairs") >= 1
    assert status_info.get("built") is True
    assert status_info.get("sent") is False
    assert status_info.get("compacted") is False
    assert status_info.get("skipped_reason") is None

    messages = [record.getMessage() for record in caplog.records]
    assert any(f"MANIFEST_AI_PACKS_UPDATED sid={sid}" in message for message in messages)

    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    from scripts import send_ai_merge_packs as send_mod

    monkeypatch.setattr(
        send_mod,
        "decide_merge_or_different",
        lambda pack, timeout: {
            "decision": "different",
            "reason": "These tradelines describe the same debt from different collectors.",
            "flags": {"account_match": False, "debt_match": False},
        },
    )
    timestamp = "2024-07-04T12:00:00Z"
    monkeypatch.setattr(send_mod, "_isoformat_timestamp", lambda dt=None: timestamp)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="scripts.send_ai_merge_packs"):
        send_ai_merge_packs_main(["--sid", sid, "--runs-root", str(runs_root)])

    messages = [record.getMessage() for record in caplog.records]
    assert any(f"MANIFEST_AI_SENT sid={sid}" in message for message in messages)
    assert any(f"MANIFEST_AI_COMPACTED sid={sid}" in message for message in messages)

    logs_path = packs_dir / "logs.txt"
    assert logs_path.exists()
    assert logs_path.read_text(encoding="utf-8")

    manifest = RunManifest.for_sid(sid)
    manifest_data = json.loads((runs_root / sid / "manifest.json").read_text(encoding="utf-8"))
    ai_packs = manifest_data["ai"]["packs"]
    assert Path(ai_packs["dir"]) == packs_dir.resolve()
    assert Path(ai_packs["index"]) == (packs_dir / "index.json").resolve()
    assert Path(ai_packs["logs"]) == logs_path.resolve()

    status_info = manifest.data.get("ai", {}).get("status", {})
    assert status_info.get("sent") is True
    assert status_info.get("compacted") is True
    assert status_info.get("skipped_reason") is None

    packs_info = manifest.data.get("ai", {}).get("packs", {})
    assert Path(packs_info.get("dir")).exists()
    assert Path(packs_info.get("index")).exists()
    assert packs_info.get("pairs") >= 1

    compact_tags_for_account(account_a)
    compact_tags_for_account(account_b)

    tags_a = json.loads((account_a / "tags.json").read_text(encoding="utf-8"))
    tags_b = json.loads((account_b / "tags.json").read_text(encoding="utf-8"))
    assert tags_a == [
        {"kind": "merge_best", "with": 16, "decision": "ai"},
        {
            "kind": "ai_decision",
            "with": 16,
            "decision": "different",
            "at": timestamp,
            "flags": {"account_match": False, "debt_match": False},
        },
        {
            "kind": "ai_resolution",
            "with": 16,
            "decision": "different",
            "flags": {"account_match": False, "debt_match": False},
            "reason": "These tradelines describe the same debt from different collectors.",
        },
    ]
    assert tags_b == [
        {"kind": "merge_best", "with": 11, "decision": "ai"},
        {
            "kind": "ai_decision",
            "with": 11,
            "decision": "different",
            "at": timestamp,
            "flags": {"account_match": False, "debt_match": False},
        },
        {
            "kind": "ai_resolution",
            "with": 11,
            "decision": "different",
            "flags": {"account_match": False, "debt_match": False},
            "reason": "These tradelines describe the same debt from different collectors.",
        },
    ]

    summary_a = json.loads((account_a / "summary.json").read_text(encoding="utf-8"))
    summary_b = json.loads((account_b / "summary.json").read_text(encoding="utf-8"))

    assert summary_a["merge_explanations"]
    assert summary_b["merge_explanations"]
    assert summary_a["ai_explanations"]
    assert summary_b["ai_explanations"]

    merge_score_a = summary_a.get("merge_scoring")
    assert merge_score_a
    assert merge_score_a["best_with"] == 16
    assert merge_score_a["score_total"] >= 0
    assert "total" in merge_score_a["reasons"]
    assert merge_score_a["matched_fields"].get("balance_owed") is True

    merge_score_b = summary_b.get("merge_scoring")
    assert merge_score_b
    assert merge_score_b["best_with"] == 11
    assert merge_score_b["matched_fields"].get("balance_owed") is True





def test_run_auto_ai_pipeline_processes_index_and_updates_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runs_root = tmp_path / "runs"
    sid, account_a, account_b = _setup_merge_case(runs_root, sid="auto-pipeline")

    monkeypatch.setattr(auto_ai, "RUNS_ROOT", runs_root)

    import backend.core.logic.merge.scorer as merge_scorer

    monkeypatch.setattr(merge_scorer, "score_bureau_pairs_cli", lambda *_, **__: None)

    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    from scripts import send_ai_merge_packs as send_mod

    timestamp = "2024-07-05T09:00:00Z"

    manifest = RunManifest.for_sid(sid)
    manifest.set_ai_enqueued()

    monkeypatch.setattr(
        send_mod,
        "decide_merge_or_different",
        lambda pack, timeout: {
            "decision": "merge",
            "reason": "Records align cleanly.",
            "flags": {"account_match": True, "debt_match": True},
        },
    )
    monkeypatch.setattr(send_mod, "_isoformat_timestamp", lambda dt=None: timestamp)

    caplog.clear()
    caplog.set_level(logging.INFO, logger="backend.pipeline.auto_ai")
    caplog.set_level(logging.INFO, logger="scripts.build_ai_merge_packs")
    caplog.set_level(logging.INFO, logger="scripts.send_ai_merge_packs")

    result = auto_ai._run_auto_ai_pipeline(sid)

    assert result == {"sid": sid, "ok": True}

    manifest = RunManifest.for_sid(sid)
    status_info = manifest.data.get("ai", {}).get("status", {})
    packs_info = manifest.data.get("ai", {}).get("packs", {})

    assert status_info.get("built") is True
    assert status_info.get("sent") is True
    assert status_info.get("compacted") is True
    assert status_info.get("skipped_reason") is None

    packs_dir = Path(packs_info.get("dir"))
    index_path = Path(packs_info.get("index"))
    assert packs_dir.exists()
    assert index_path.exists()
    assert packs_info.get("pairs") >= 1

    tags_a = json.loads((account_a / "tags.json").read_text(encoding="utf-8"))
    tags_b = json.loads((account_b / "tags.json").read_text(encoding="utf-8"))
    assert any(tag.get("kind") == "ai_decision" for tag in tags_a)
    assert any(tag.get("kind") == "ai_decision" for tag in tags_b)

    messages = [record.getMessage() for record in caplog.records]
    assert any(f"AUTO_AI_PACKS_FOUND sid={sid}" in message for message in messages)
    assert any(f"MANIFEST_AI_PACKS_UPDATED sid={sid}" in message for message in messages)
    assert any(f"MANIFEST_AI_SENT sid={sid}" in message for message in messages)
    assert any(f"MANIFEST_AI_COMPACTED sid={sid}" in message for message in messages)


def test_run_auto_ai_pipeline_skips_when_index_missing(monkeypatch, tmp_path):
    runs_root = tmp_path / 'runs'
    monkeypatch.setattr(auto_ai, 'RUNS_ROOT', runs_root)
    sid, account_a, account_b = _setup_merge_case(runs_root, sid='missing-index')

    import backend.core.logic.merge.scorer as merge_scorer
    import backend.core.logic.ai.send_ai_merge_packs as send_module

    monkeypatch.setattr(merge_scorer, 'score_bureau_pairs_cli', lambda *_, **__: None)

    def fake_build_cli(sid_value, runs_root_value):
        packs_dir = auto_ai.packs_dir_for(sid_value, runs_root=runs_root_value)
        packs_dir.mkdir(parents=True, exist_ok=True)
        # intentionally do not create index.json to trigger no_packs path

    monkeypatch.setattr(auto_ai, '_build_ai_packs', fake_build_cli)

    def fake_send(*args, **kwargs):  # pragma: no cover - ensure not called
        raise AssertionError('send should not run when index is missing')

    monkeypatch.setattr(send_module, 'run_send_for_sid', fake_send)

    manifest_before = RunManifest.for_sid(sid)

    result = auto_ai._run_auto_ai_pipeline(sid)

    assert result == {'sid': sid, 'skipped': 'no_packs'}

    manifest_after = RunManifest.for_sid(sid)
    status = manifest_after.data.get('ai', {}).get('status', {})
    assert status.get('skipped_reason') == 'no_packs'
    assert status.get('sent') is False
    assert status.get('compacted') is False

