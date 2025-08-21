import json
import threading
import time

from backend.analytics.analytics_tracker import get_counters, reset_counters
from backend.analytics.batch_exporter import export_accounts


def test_backpressure_and_metrics(tmp_path):
    reset_counters()
    active = 0
    max_active = 0
    lock = threading.Lock()

    def worker(payload):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return payload["val"] * 2

    accounts = [{"val": i} for i in range(5)]
    dlq_dir = tmp_path / "dlq"
    results = export_accounts(
        accounts,
        worker,
        chunk_size=1,
        max_workers=5,
        output_queue_max=2,
        dlq_dir=dlq_dir,
    )
    assert sorted(results) == [i * 2 for i in range(5)]
    assert max_active <= 2

    counters = get_counters()
    assert counters["batch.jobs_total"] == 5
    assert counters["batch.failures_total"] == 0
    assert counters["batch.records_exported"] == 5
    assert "batch.duration_ms" in counters


def test_dlq_on_failure(tmp_path):
    reset_counters()

    def worker(payload):
        if payload["val"] == 2:
            raise ValueError("boom")
        return payload["val"]

    accounts = [{"val": i} for i in range(3)]
    dlq_dir = tmp_path / "dlq"
    results = export_accounts(
        accounts,
        worker,
        chunk_size=1,
        max_workers=2,
        output_queue_max=2,
        dlq_dir=dlq_dir,
    )
    assert sorted(results) == [0, 1]

    counters = get_counters()
    assert counters["batch.jobs_total"] == 3
    assert counters["batch.failures_total"] == 1
    assert counters["batch.records_exported"] == 2

    files = list(dlq_dir.glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["val"] == 2
