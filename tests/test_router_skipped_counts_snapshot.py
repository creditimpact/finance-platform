import json
from pathlib import Path

from backend.analytics.analytics_tracker import (
    emit_counter,
    reset_counters,
    save_analytics_snapshot,
)


def test_snapshot_records_router_skipped(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    reset_counters()
    emit_counter("router.skipped.ignore")
    emit_counter("router.skipped.paydown_first", 2)
    save_analytics_snapshot({}, {})
    file = next(Path("analytics_data").glob("*.json"))
    data = json.loads(file.read_text())
    skipped = data["metrics"]["router_skipped"]
    assert skipped["ignore"] == 1
    assert skipped["paydown_first"] == 2
