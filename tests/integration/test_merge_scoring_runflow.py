from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


def _write_bureaus(runs_root: Path, sid: str, idx: int, payload: dict[str, object]) -> None:
    account_dir = runs_root / sid / "cases" / "accounts" / str(idx)
    account_dir.mkdir(parents=True, exist_ok=True)
    (account_dir / "bureaus.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_merge_scoring_runflow_handles_masked_accounts(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "SID-merge-scoring"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")

    fake_requests = types.ModuleType("requests")
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    import backend.core.runflow as runflow
    import backend.core.logic.report_analysis.account_merge as account_merge

    importlib.reload(runflow)
    importlib.reload(account_merge)

    _write_bureaus(
        runs_root,
        sid,
        0,
        {
            "transunion": {"account_number_display": "****1234"},
            "equifax": {"account_number_display": "XXXX1234"},
        },
    )
    _write_bureaus(
        runs_root,
        sid,
        1,
        {
            "transunion": {"account_number_display": "123456789012"},
            "equifax": {"account_number_display": "****9012"},
        },
    )
    _write_bureaus(
        runs_root,
        sid,
        2,
        {
            "transunion": {"account_number_display": "N/A"},
            "experian": {"account_number_display": "—"},
        },
    )

    scores = account_merge.score_all_pairs_0_100(sid, [], runs_root=runs_root)

    expected_raw_values = {
        (0, 1): {"a": "****1234", "b": "123456789012"},
        (0, 2): {"a": "****1234", "b": "N/A"},
        (1, 2): {"a": "123456789012", "b": "N/A"},
    }

    for left, right in expected_raw_values:
        aux = (
            scores.get(left, {})
            .get(right, {})
            .get("aux", {})
            .get("account_number", {})
        )
        assert aux.get("raw_values") == expected_raw_values[(left, right)]
        assert not aux.get("candidate_map")

    events_path = runs_root / sid / "runflow_events.jsonl"
    assert events_path.exists()

    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    steps = [event.get("step") for event in events]

    assert any(event.get("step") == "merge_scoring" and event.get("status") == "start" for event in events)
    assert any(event.get("step") == "merge_scoring" and event.get("status") == "success" for event in events)

    pack_skips = [event for event in events if event.get("step") == "pack_skip"]
    assert {event.get("account") for event in pack_skips} == {"0-1", "0-2", "1-2"}
    for skip in pack_skips:
        assert skip.get("out", {}).get("reason") == "no_candidates"

    steps_path = runs_root / sid / "runflow_steps.json"
    steps_payload = json.loads(steps_path.read_text(encoding="utf-8"))
    merge_stage = steps_payload["stages"]["merge"]
    stage_steps = merge_stage["steps"]
    summary = merge_stage.get("summary", {})
    step_names = [entry.get("name") for entry in stage_steps]
    assert step_names
    assert step_names[0] == "merge_scoring_start"
    assert step_names[-1] == "merge_scoring_finish"
    allowed_steps = {
        "merge_scoring_start",
        "acctnum_normalize",
        "pack_create",
        "merge_scoring_finish",
    }
    assert all(name in allowed_steps for name in step_names)

    pack_entries = [entry for entry in stage_steps if entry.get("name") == "pack_create"]
    normalize_entries = [
        entry for entry in stage_steps if entry.get("name") == "acctnum_normalize"
    ]

    assert isinstance(summary, dict)
    summary_present = bool(summary)
    if summary_present:
        expected_keys = {"created_packs", "scored_pairs", "empty_ok"}
        if "packs_created" in summary:
            expected_keys.add("packs_created")
        assert expected_keys.issubset(set(summary))
        created_packs = int(summary.get("created_packs", len(pack_entries)) or 0)
        packs_created_raw = summary.get("packs_created")
        if packs_created_raw is not None:
            packs_created = int(packs_created_raw or 0)
            assert packs_created == len(pack_entries)
        else:
            packs_created = created_packs
        scored_pairs_summary = int(summary.get("scored_pairs", 0) or 0)
        summary_empty_ok = summary.get("empty_ok")
    else:
        created_packs = len(pack_entries)
        packs_created = created_packs
        scored_pairs_summary = None
        summary_empty_ok = None

    if created_packs == 0:
        assert not pack_entries
        stage_empty_ok = merge_stage.get("empty_ok")
        if summary_empty_ok is not None:
            assert summary_empty_ok is True
        if stage_empty_ok is not None:
            assert stage_empty_ok is True
    else:
        assert pack_entries
        assert all(entry.get("status") == "success" for entry in pack_entries)
        assert all(entry.get("out", {}).get("path") for entry in pack_entries)
        if summary_empty_ok is not None:
            assert summary_empty_ok is False
        stage_empty_ok = merge_stage.get("empty_ok")
        if stage_empty_ok is not None:
            assert stage_empty_ok is False

    if normalize_entries:
        assert len(normalize_entries) == 1
        assert normalize_entries[0].get("status") == "success"

    pairs_index_path = runs_root / sid / "ai_packs" / "merge" / "pairs_index.json"
    index_payload = json.loads(pairs_index_path.read_text(encoding="utf-8"))
    if scored_pairs_summary is not None:
        assert scored_pairs_summary == index_payload["totals"]["scored_pairs"]
    assert len(index_payload["pairs"]) == 3


def test_merge_runflow_steps_single_pack(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "SID-merge-single-pack"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")

    fake_requests = types.ModuleType("requests")
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    import backend.core.runflow as runflow
    import backend.core.logic.report_analysis.account_merge as account_merge

    importlib.reload(runflow)
    importlib.reload(account_merge)

    _write_bureaus(runs_root, sid, 0, {"transunion": {"account_number_display": "1111"}})
    _write_bureaus(runs_root, sid, 1, {"experian": {"account_number_display": "1111"}})

    call_counter = {"count": 0}

    def fake_score_pair(_left, _right, _cfg):
        call_counter["count"] += 1
        return {
            "total": 99,
            "parts": {"balance_owed": 99},
            "aux": {
                "acctnum_level": "strong",
                "account_number": {
                    "raw_values": {"a": "1111", "b": "1111"},
                    "acctnum_debug": {"short": "match", "long": "match", "why": "strong"},
                },
            },
            "triggers": ["strong:acctnum"],
            "trigger_events": [],
            "conflicts": [],
            "decision": "ai",
        }

    def fake_build_ai_pack_for_pair(sid_arg, runs_root_arg, left, right, highlights):
        merge_paths = account_merge.get_merge_paths(Path(runs_root_arg), sid_arg, create=True)
        pack_path = account_merge.pair_pack_path(merge_paths, left, right)
        pack_path.parent.mkdir(parents=True, exist_ok=True)
        pack_path.write_text("{}\n", encoding="utf-8")
        return {"pair": {"a": left, "b": right}, "highlights": highlights}

    monkeypatch.setattr(account_merge, "score_pair_0_100", fake_score_pair)
    monkeypatch.setattr(account_merge, "build_ai_pack_for_pair", fake_build_ai_pack_for_pair)

    account_merge.score_all_pairs_0_100(sid, [0, 1], runs_root=runs_root)

    assert call_counter["count"] == 1

    steps_path = runs_root / sid / "runflow_steps.json"
    steps_payload = json.loads(steps_path.read_text(encoding="utf-8"))
    merge_stage = steps_payload["stages"]["merge"]
    stage_steps = merge_stage["steps"]
    step_names = [entry.get("name") for entry in stage_steps]
    assert step_names[0] == "merge_scoring_start"
    assert step_names[-1] == "merge_scoring_finish"
    assert all(
        name in {"merge_scoring_start", "acctnum_normalize", "pack_create", "merge_scoring_finish"}
        for name in step_names
    )

    pack_entries = [entry for entry in stage_steps if entry.get("name") == "pack_create"]

    assert len(pack_entries) == 1
    pack_out = pack_entries[0].get("out", {})
    assert pack_out.get("path")

    summary = merge_stage.get("summary", {})
    packs_created_summary = summary.get("packs_created")
    if packs_created_summary is not None:
        assert int(packs_created_summary or 0) == len(pack_entries)
    created_packs_summary = summary.get("created_packs")
    if created_packs_summary is not None:
        assert int(created_packs_summary or 0) == len(pack_entries)


def test_merge_runflow_steps_no_packs(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    sid = "SID-merge-no-packs"

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("RUNFLOW_EVENTS", "1")
    monkeypatch.setenv("RUNFLOW_VERBOSE", "1")

    fake_requests = types.ModuleType("requests")
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    import backend.core.runflow as runflow
    import backend.core.logic.report_analysis.account_merge as account_merge

    importlib.reload(runflow)
    importlib.reload(account_merge)

    _write_bureaus(runs_root, sid, 0, {"transunion": {"account_number_display": "2222"}})
    _write_bureaus(runs_root, sid, 1, {"experian": {"account_number_display": "3333"}})

    def fake_score_pair(_left, _right, _cfg):
        return {
            "total": 0,
            "parts": {"balance_owed": 0},
            "aux": {
                "acctnum_level": "none",
                "account_number": {
                    "raw_values": {"a": "2222", "b": "3333"},
                    "acctnum_debug": {"short": "", "long": "", "why": "different"},
                },
            },
            "triggers": [],
            "trigger_events": [],
            "conflicts": [],
            "decision": "different",
        }

    monkeypatch.setattr(account_merge, "score_pair_0_100", fake_score_pair)

    account_merge.score_all_pairs_0_100(sid, [0, 1], runs_root=runs_root)

    runflow.runflow_end_stage(
        sid,
        "merge",
        status="success",
        summary={"created_packs": 0, "packs_created": 0, "scored_pairs": 1, "empty_ok": True},
        stage_status="empty",
        empty_ok=True,
    )

    steps_path = runs_root / sid / "runflow_steps.json"
    steps_payload = json.loads(steps_path.read_text(encoding="utf-8"))
    merge_stage = steps_payload["stages"]["merge"]
    stage_steps = merge_stage["steps"]

    pack_entries = [entry for entry in stage_steps if entry.get("name") == "pack_create"]
    step_names = [entry.get("name") for entry in stage_steps]

    assert not pack_entries
    assert step_names[0] == "merge_scoring_start"
    assert step_names[-1] == "merge_scoring_finish"
    assert all(
        name in {"merge_scoring_start", "acctnum_normalize", "merge_scoring_finish"}
        for name in step_names
    )

    summary = merge_stage.get("summary", {})
    assert summary.get("created_packs") == 0
    assert summary.get("packs_created") == 0
    assert summary.get("empty_ok") is True
    assert merge_stage.get("empty_ok") is True
