import json
from pathlib import Path

from backend.audit.audit import AuditLogger
from backend.core.logic.constants import FallbackReason
from backend.audit.trace_exporter import export_trace_breakdown


def _build_sample_audit() -> AuditLogger:
    audit = AuditLogger()
    audit.data["session_id"] = "sess1"
    audit.log_account(
        "1",
        {
            "stage": "strategy_decision",
            "action": "dispute",
            "recommended_action": "Dispute",
            "reason": "Incorrect info",
        },
    )
    audit.log_account(
        "1",
        {
            "stage": "strategy_fallback",
            "fallback_reason": FallbackReason.UNRECOGNIZED_TAG.value,
            "strategist_action": "remove",
        },
    )
    return audit


def _build_strategy() -> dict:
    return {
        "accounts": [
            {
                "account_id": "1",
                "name": "Acct 1",
                "recommendation": {
                    "action_tag": "dispute",
                    "recommended_action": "Dispute",
                    "advisor_comment": "Incorrect info",
                },
            }
        ]
    }


def test_export_trace_breakdown_files_exist(tmp_path: Path) -> None:
    audit = _build_sample_audit()
    strategy = _build_strategy()
    export_trace_breakdown(audit, strategy, strategy["accounts"], tmp_path)

    base = tmp_path / audit.data["session_id"] / "trace"
    names = {
        "strategist_raw_output.json",
        "strategy_decision.json",
        "fallback_reason.json",
        "recommendation_summary.json",
    }
    for name in names:
        file_path = base / name
        assert file_path.exists(), f"missing {name}"
        data = json.loads(file_path.read_text())
        assert data["session_id"] == "sess1"
        assert "run_date" in data
        assert "1" in data.get("accounts", {})


def test_export_trace_breakdown_strategy_vs_fallback_consistency(
    tmp_path: Path,
) -> None:
    audit = _build_sample_audit()
    strategy = _build_strategy()
    export_trace_breakdown(audit, strategy, strategy["accounts"], tmp_path)

    base = tmp_path / audit.data["session_id"] / "trace"
    decisions = json.loads((base / "strategy_decision.json").read_text())
    fallbacks = json.loads((base / "fallback_reason.json").read_text())

    assert decisions["accounts"]["1"]["source"] == "fallback"
    assert "1" in fallbacks["accounts"]
