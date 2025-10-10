"""Helpers for building lightweight frontend validation packs."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from backend.core.io.json_io import _atomic_write_json

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _default_runs_root() -> Path:
    root_env = os.getenv("RUNS_ROOT")
    return Path(root_env) if root_env else Path("runs")


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        return _default_runs_root()
    return Path(runs_root)


def _strip_raw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_raw(val)
            for key, val in value.items()
            if str(key) != "raw"
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_strip_raw(item) for item in value]
    return value


def _sanitize_findings(value: Any) -> list[MutableMapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []

    sanitized: list[MutableMapping[str, Any]] = []
    for entry in value:
        if isinstance(entry, Mapping):
            sanitized.append(dict(_strip_raw(entry)))
    return sanitized


def _account_sort_key(path: Path) -> tuple[int, Any]:
    name = path.name
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def generate_frontend_packs_for_run(
    sid: str,
    *,
    runs_root: Path | str | None = None,
) -> dict[str, Any]:
    """Aggregate validation findings into a frontend-friendly payload."""

    base_root = _resolve_runs_root(runs_root)
    run_dir = base_root / sid
    accounts_dir = run_dir / "cases" / "accounts"
    frontend_dir = run_dir / "frontend"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    output_path = frontend_dir / "packs.json"

    accounts_payload: list[dict[str, Any]] = []
    total_findings = 0

    if accounts_dir.is_dir():
        account_dirs = [p for p in accounts_dir.iterdir() if p.is_dir()]
        for account_dir in sorted(account_dirs, key=_account_sort_key):
            summary_path = account_dir / "summary.json"
            if not summary_path.exists():
                continue

            try:
                raw_summary = summary_path.read_text(encoding="utf-8")
                summary = json.loads(raw_summary)
            except Exception:  # pragma: no cover - defensive logging
                log.warning(
                    "FRONTEND_PACK_SUMMARY_READ_FAILED sid=%s path=%s",
                    sid,
                    summary_path,
                    exc_info=True,
                )
                continue

            validation_block = summary.get("validation_requirements")
            findings_value = None
            if isinstance(validation_block, Mapping):
                findings_value = validation_block.get("findings")

            findings = _sanitize_findings(findings_value)
            if not findings:
                continue

            account_entry: dict[str, Any] = {
                "account_id": str(summary.get("account_id") or account_dir.name),
                "account_key": account_dir.name,
                "findings": findings,
                "summary_path": str(summary_path),
            }
            account_index = summary.get("account_index")
            if account_index is not None:
                account_entry["account_index"] = account_index

            accounts_payload.append(account_entry)
            total_findings += len(findings)

    payload = {
        "sid": sid,
        "generated_at": _now_iso(),
        "accounts": accounts_payload,
        "accounts_with_findings": len(accounts_payload),
        "packs_count": total_findings,
    }

    _atomic_write_json(output_path, payload)

    log.info(
        "FRONTEND_PACKS_GENERATED sid=%s accounts=%d findings=%d path=%s",
        sid,
        len(accounts_payload),
        total_findings,
        output_path,
    )

    return {
        "sid": sid,
        "packs_count": total_findings,
        "accounts_with_findings": len(accounts_payload),
        "path": output_path,
    }


__all__ = ["generate_frontend_packs_for_run"]
