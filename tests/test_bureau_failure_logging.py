import json
import logging

from backend.core.logic.report_analysis.report_prompting import log_bureau_failure


def test_log_bureau_failure_emits_json(caplog):
    with caplog.at_level(logging.INFO, logger="backend.audit.audit"):
        log_bureau_failure(
            error_code="BROKEN_JSON",
            bureau="Experian",
            expected_headings=3,
            found_accounts=1,
            tokens=42,
            latency=123.4,
        )
    assert caplog.records, "No log records captured"
    event, payload = caplog.records[0].message.split(" ", 1)
    assert event == "bureau_failure"
    data = json.loads(payload)
    assert data == {
        "error_code": "BROKEN_JSON",
        "bureau": "Experian",
        "expected_headings": 3,
        "found_accounts": 1,
        "tokens": 42,
        "latency": 123.4,
    }
