import json
import sqlite3
from pathlib import Path

from backend.analytics.batch_runner import BatchFilters, BatchRunner


def test_batch_runner_run(tmp_path):
    samples = Path(__file__).parent / "helpers" / "batch_samples.json"

    def fake_fetch(self, filters):
        return json.loads(samples.read_text())

    runner = BatchRunner(job_store=tmp_path / "jobs.sqlite", output_dir=tmp_path)
    runner._fetch_samples = fake_fetch.__get__(runner, BatchRunner)

    filters = BatchFilters(action_tags=["fraud_dispute"])
    job_id = runner.run(filters, format="json")

    json_path = tmp_path / f"{job_id}.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert "missing_fields" in data

    # idempotent run
    job_id2 = runner.run(filters, format="json")
    assert job_id2 == job_id

    with sqlite3.connect(tmp_path / "jobs.sqlite") as conn:
        cur = conn.execute("SELECT COUNT(*) FROM batch_jobs")
        assert cur.fetchone()[0] == 1

    # retry should reprocess and keep single record
    runner.retry(job_id)
    with sqlite3.connect(tmp_path / "jobs.sqlite") as conn:
        cur = conn.execute("SELECT COUNT(*) FROM batch_jobs")
        assert cur.fetchone()[0] == 1

