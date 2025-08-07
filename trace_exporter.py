import json
from pathlib import Path
from typing import Any, Dict


def export_trace_file(audit: Any, session_id: str) -> Path:
    """Export strategist and fallback diagnostics to trace.json.

    Parameters
    ----------
    audit: AuditLogger or dict
        The audit object or its underlying data structure.
    session_id: str
        Current client session identifier.
    """
    data: Dict[str, Any] = audit.data if hasattr(audit, "data") else audit

    steps = data.get("steps", [])
    accounts = data.get("accounts", {})

    strategist_raw_output = ""
    for step in steps:
        if step.get("stage") == "strategist_raw_output":
            strategist_raw_output = step.get("details", {}).get("content", "")
            break

    strategist_failure_reasons = [
        step.get("details", {}).get("failure_reason")
        for step in steps
        if step.get("stage") == "strategist_failure"
    ]

    strategy_decision_log = []
    fallback_actions = []
    per_account_failures = []
    recommendation_summary = []

    for acc_id, entries in accounts.items():
        for entry in entries:
            stage = entry.get("stage")
            if stage == "strategy_decision":
                decision = {"account_id": acc_id}
                for key in [
                    "action",
                    "recommended_action",
                    "flags",
                    "reason",
                    "classification",
                ]:
                    if entry.get(key) is not None:
                        decision[key] = entry.get(key)
                strategy_decision_log.append(decision)
                recommendation_summary.append(
                    {
                        "account_id": acc_id,
                        "action": entry.get("action"),
                        "recommended_action": entry.get("recommended_action"),
                    }
                )
            elif stage == "strategy_fallback":
                fb = {"account_id": acc_id}
                for key in [
                    "fallback_reason",
                    "strategist_action",
                    "overrode_strategist",
                    "failure_reason",
                    "raw_action",
                ]:
                    if entry.get(key) is not None:
                        fb[key] = entry.get(key)
                fallback_actions.append(fb)
                fail_info = {
                    k: entry.get(k)
                    for k in ("failure_reason", "fallback_reason")
                    if entry.get(k) is not None
                }
                if fail_info:
                    per_account_failures.append({"account_id": acc_id, **fail_info})

    trace: Dict[str, Any] = {
        "strategist_raw_output": strategist_raw_output,
        "strategist_failure_reasons": strategist_failure_reasons,
        "strategy_decision_log": strategy_decision_log,
        "fallback_actions": fallback_actions,
        "per_account_failures": per_account_failures,
        "recommendation_summary": recommendation_summary,
    }

    trace_folder = Path("client_output") / session_id / "trace"
    trace_folder.mkdir(parents=True, exist_ok=True)
    trace_path = trace_folder / "trace.json"
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    return trace_path

__all__ = ["export_trace_file"]
