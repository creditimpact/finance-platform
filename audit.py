from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_current_audit: Optional["AuditLogger"] = None

class AuditLogger:
    """Collects structured audit information for a credit repair run."""

    def __init__(self) -> None:
        self.data: Dict[str, Any] = {
            "start_time": datetime.utcnow().isoformat(),
            "steps": [],
            "accounts": {},
            "errors": [],
        }

    def log_step(self, stage: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.data["steps"].append(
            {
                "stage": stage,
                "timestamp": datetime.utcnow().isoformat(),
                "details": details or {},
            }
        )

    def log_account(self, account_id: Any, info: Dict[str, Any]) -> None:
        acc = self.data["accounts"].setdefault(str(account_id), [])
        entry = {"timestamp": datetime.utcnow().isoformat()}
        entry.update(info)
        acc.append(entry)

    def log_error(self, message: str) -> None:
        self.data["errors"].append(
            {"timestamp": datetime.utcnow().isoformat(), "message": message}
        )

    def save(self, folder: Path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "audit.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        return path


def start_audit() -> AuditLogger:
    global _current_audit
    _current_audit = AuditLogger()
    return _current_audit


def get_audit() -> Optional[AuditLogger]:
    return _current_audit


def clear_audit() -> None:
    global _current_audit
    _current_audit = None
