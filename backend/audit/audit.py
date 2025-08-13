from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class AuditLevel(Enum):
    ESSENTIAL = 1
    VERBOSE = 2


class AuditLogger:
    """Collects structured audit information for a credit repair run."""

    ESSENTIAL_STEPS = {
        "strategist_invocation",
        "strategist_raw_output",
        "strategist_failure",
        "strategy_generated",
        "strategy_merged",
        "strategy_decision",
        "pre_strategy_fallback",
        "strategy_fallback",
    }

    def __init__(self, level: AuditLevel = AuditLevel.ESSENTIAL) -> None:
        self.level = level
        self.data: Dict[str, Any] = {
            "start_time": datetime.now(UTC).isoformat(),
            "steps": [],
            "accounts": {},
            "errors": [],
        }

    def log_step(self, stage: str, details: Optional[Dict[str, Any]] = None) -> None:
        if self.level == AuditLevel.ESSENTIAL and stage not in self.ESSENTIAL_STEPS:
            return
        self.data["steps"].append(
            {
                "stage": stage,
                "timestamp": datetime.now(UTC).isoformat(),
                "details": details or {},
            }
        )

    def log_account(self, account_id: Any, info: Dict[str, Any]) -> None:
        if (
            self.level == AuditLevel.ESSENTIAL
            and info.get("stage") not in self.ESSENTIAL_STEPS
        ):
            return
        acc = self.data["accounts"].setdefault(str(account_id), [])
        entry = {"timestamp": datetime.now(UTC).isoformat()}
        entry.update(info)
        acc.append(entry)

    def log_error(self, message: str) -> None:
        self.data["errors"].append(
            {"timestamp": datetime.now(UTC).isoformat(), "message": message}
        )

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "audit.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        return path


def create_audit_logger(
    session_id: str, level: AuditLevel = AuditLevel.ESSENTIAL
) -> AuditLogger:
    audit = AuditLogger(level=level)
    audit.data["session_id"] = session_id
    return audit
