"""Orchestrate Validation AI pack creation and inference for a case run."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.config import ENABLE_VALIDATION_AI

from .build_packs import build_validation_packs, resolve_manifest_paths
from .send_packs import send_validation_packs


def _read_manifest(manifest_path: Path | str) -> Mapping[str, Any]:
    manifest_text = Path(manifest_path).read_text(encoding="utf-8")
    return json.loads(manifest_text)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _append_log(log_path: Path, event: str, **payload: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": _utc_now(), "event": event, **payload}
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(serialized + "\n")


def run_case(manifest_path: Path | str) -> Mapping[str, Any]:
    """Execute the validation AI build/send flow for ``manifest_path``."""

    manifest = _read_manifest(manifest_path)
    paths = resolve_manifest_paths(manifest)

    if not ENABLE_VALIDATION_AI:
        _append_log(paths.log_path, "validation_ai_skipped", reason="disabled")
        return {"enabled": False, "reason": "validation_ai_disabled"}

    _append_log(paths.log_path, "validation_ai_start", sid=paths.sid)
    try:
        build_items = build_validation_packs(manifest_path)
        _append_log(
            paths.log_path,
            "validation_ai_built",
            sid=paths.sid,
            accounts=len(build_items),
        )

        send_results = send_validation_packs(manifest_path)
        _append_log(
            paths.log_path,
            "validation_ai_sent",
            sid=paths.sid,
            accounts=len(send_results),
        )
    except Exception as exc:
        _append_log(
            paths.log_path,
            "validation_ai_error",
            sid=paths.sid,
            error=str(exc),
        )
        raise

    return {
        "enabled": True,
        "build": build_items,
        "send": send_results,
    }


__all__ = ["run_case"]
