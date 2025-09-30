import json
import os

from backend.core.config import ENABLE_VALIDATION_REQUIREMENTS, VALIDATION_DEBUG
from backend.core.logic.validation_requirements import (
    build_validation_requirements_for_account,
)


def run(account_dir: str) -> dict:
    """
    Runs the validation-requirements builder on a single account folder.
    Writes back into summary.json (under 'validation_requirements').
    Idempotent: safe to run multiple times.
    """
    if not ENABLE_VALIDATION_REQUIREMENTS:
        return {"skipped": True, "reason": "flag_off"}

    bureaus_json = os.path.join(account_dir, "bureaus.json")
    summary_json = os.path.join(account_dir, "summary.json")
    if not (os.path.isfile(bureaus_json) and os.path.isfile(summary_json)):
        return {"skipped": True, "reason": "missing_inputs"}

    req = build_validation_requirements_for_account(account_dir) or {}

    try:
        with open(summary_json, "r+", encoding="utf-8") as f:
            summary = json.load(f)
            summary["validation_requirements"] = req
            if not VALIDATION_DEBUG and "validation_debug" in summary:
                summary.pop("validation_debug", None)
            f.seek(0)
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.truncate()
    except Exception as exc:  # pragma: no cover - defensive logging path
        return {"skipped": True, "reason": f"write_error:{exc}"}

    return {"skipped": False, "requirements_count": req.get("count")}
