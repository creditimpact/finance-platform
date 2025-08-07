import csv
import json
from datetime import datetime, timedelta, UTC
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from logic import outcomes_store
from scripts import export_outcomes


def test_record_and_filter(tmp_path, monkeypatch):
    file_path = tmp_path / "outcomes.json"
    monkeypatch.setattr(outcomes_store, "OUTCOMES_FILE", str(file_path))

    outcomes_store.record_outcome("sess1", "acc1", "Equifax", 1, "deleted", 3, "error")
    outcomes_store.record_outcome("sess2", "acc2", "Experian", 1, "verified")

    assert file_path.exists()
    data = json.loads(file_path.read_text())
    assert len(data) == 2

    subset = outcomes_store.get_outcomes({"result": "deleted"})
    assert len(subset) == 1
    assert subset[0]["account_id"] == "acc1"


def test_export_last_week(tmp_path, monkeypatch):
    file_path = tmp_path / "outcomes.json"
    monkeypatch.setattr(outcomes_store, "OUTCOMES_FILE", str(file_path))

    outcomes_store.record_outcome("sess1", "acc1", "Equifax", 1, "deleted")
    outcomes_store.record_outcome("sess2", "acc2", "Experian", 1, "verified")

    data = outcomes_store.get_outcomes()
    data[0]["timestamp"] = (datetime.now(UTC) - timedelta(days=8)).isoformat()
    file_path.write_text(json.dumps(data))

    export_dir = tmp_path / "exports"
    monkeypatch.setattr(export_outcomes, "EXPORT_DIR", export_dir)
    export_outcomes.export_outcomes()

    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    json_path = export_dir / f"outcomes_{date_str}.json"
    csv_path = export_dir / f"outcomes_{date_str}.csv"
    assert json_path.exists()
    assert csv_path.exists()

    exported = json.loads(json_path.read_text())
    assert len(exported) == 1
    assert exported[0]["account_id"] == "acc2"

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["account_id"] == "acc2"
