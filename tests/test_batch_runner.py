import json
from pathlib import Path

from backend.analytics.batch_runner import run_staging_batch


def test_run_staging_batch(tmp_path):
    samples = Path(__file__).parent / "helpers" / "batch_samples.json"
    report = run_staging_batch(samples, limit=2, output_dir=tmp_path)
    assert "finalization_pass_rate" in report
    assert "sanitizer" in report
    json_files = list(tmp_path.glob("*.json"))
    csv_files = list(tmp_path.glob("*.csv"))
    assert len(json_files) == 1
    assert len(csv_files) == 1
    data = json.loads(json_files[0].read_text())
    assert "missing_fields" in data
