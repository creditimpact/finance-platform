from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path


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

    assert "merge_scoring_start" in steps
    assert "merge_scoring_finish" in steps

    pack_skips = [event for event in events if event.get("step") == "pack_skip"]
    assert {event.get("account") for event in pack_skips} == {"0-1", "0-2", "1-2"}
    for skip in pack_skips:
        assert skip.get("out", {}).get("reason") == "no_candidates"
