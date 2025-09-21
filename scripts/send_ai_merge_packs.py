"""Send merge V2 AI packs to the adjudicator service."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

try:  # pragma: no cover - convenience bootstrap for direct execution
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback when bootstrap is unavailable
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from backend.config import AI_REQUEST_TIMEOUT
from backend.core.ai.adjudicator import decide_merge_or_different
from backend.core.io.tags import upsert_tag
from backend.pipeline.runs import RunManifest, persist_manifest

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SCHEDULE = "1,3,7"


def _load_index(path: Path) -> list[Mapping[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Pack index must be a list: {path}")
    return [dict(entry) for entry in data]


def _load_pack(path: Path) -> Mapping[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, MappingABC):
        raise ValueError(f"AI pack must be a JSON object: {path}")
    return data


def _env_max_retries() -> int:
    raw = os.getenv("AI_MAX_RETRIES")
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_RETRIES
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("AI_MAX_RETRIES must be an integer") from exc
    return max(0, value)


def _parse_backoff_schedule(raw: str | None) -> list[float]:
    if raw is None:
        return []
    parts = [part.strip() for part in raw.split(",")]
    schedule: list[float] = []
    for part in parts:
        if not part:
            continue
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError("AI_BACKOFF_SCHEDULE values must be numbers") from exc
        if value < 0:
            raise ValueError("AI_BACKOFF_SCHEDULE values must be non-negative")
        schedule.append(value)
    return schedule


def _env_backoff_schedule() -> list[float]:
    raw = os.getenv("AI_BACKOFF_SCHEDULE")
    if raw is None:
        raw = DEFAULT_BACKOFF_SCHEDULE
    return _parse_backoff_schedule(raw)


def _backoff_delay(schedule: Sequence[float], attempt_number: int) -> float:
    if not schedule:
        return 0.0
    index = min(max(attempt_number - 1, 0), len(schedule) - 1)
    return schedule[index]


def _resolve_packs_dir(runs_root: Path, sid: str, override: str | None) -> Path:
    if override:
        return Path(override)
    return runs_root / sid / "ai_packs"


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _log_factory(path: Path, sid: str, pair: Mapping[str, int], file_name: str):
    def _log(event: str, payload: Mapping[str, object] | None = None) -> None:
        extras: dict[str, object] = {
            "sid": sid,
            "pair": {"a": pair["a"], "b": pair["b"]},
            "file": file_name,
        }
        if payload:
            extras.update(payload)
        serialized = json.dumps(extras, ensure_ascii=False, sort_keys=True)
        line = f"{_isoformat_timestamp()} AI_ADJUDICATOR_{event} {serialized}\n"
        _append_log(path, line)

    return _log


def _isoformat_timestamp(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _accounts_root(runs_root: Path, sid: str) -> Path:
    return runs_root / sid / "cases" / "accounts"


def _ensure_int(value: object, label: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{label} must be an integer") from exc


def _should_mark_same_debt(payload: Mapping[str, object]) -> bool:
    decision = str(payload.get("decision") or "")
    if decision.strip().lower() == "same_debt":
        return True

    reason = str(payload.get("reason") or "")
    if "same debt" in reason.lower():
        return True

    flags = payload.get("flags")
    if isinstance(flags, MappingABC):
        flag_value = flags.get("same_debt")
        if isinstance(flag_value, bool) and flag_value:
            return True
    return False


def _write_decision_tags(
    runs_root: Path,
    sid: str,
    a_idx: int,
    b_idx: int,
    decision: str,
    reason: str,
    timestamp: str,
    payload: Mapping[str, object],
) -> None:
    base = _accounts_root(runs_root, sid)
    same_debt = _should_mark_same_debt(payload)

    for source_idx, other_idx in ((a_idx, b_idx), (b_idx, a_idx)):
        tag_path = base / str(source_idx) / "tags.json"
        decision_tag = {
            "kind": "ai_decision",
            "tag": "ai_decision",
            "source": "ai_adjudicator",
            "with": other_idx,
            "decision": decision,
            "reason": reason,
            "at": timestamp,
        }
        upsert_tag(tag_path, decision_tag, unique_keys=("kind", "with", "source"))

        if same_debt:
            same_debt_tag = {
                "kind": "same_debt_pair",
                "with": other_idx,
                "source": "ai_adjudicator",
                "reason": reason,
                "at": timestamp,
            }
            upsert_tag(tag_path, same_debt_tag, unique_keys=("kind", "with", "source"))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True, help="Session identifier")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory containing runs/<SID> outputs",
    )
    parser.add_argument(
        "--packs-dir",
        default=None,
        help="Optional override for the directory containing ai packs",
    )
    args = parser.parse_args(argv)

    sid = str(args.sid)
    runs_root = Path(args.runs_root)
    packs_dir = _resolve_packs_dir(runs_root, sid, args.packs_dir)
    index_path = packs_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Pack index not found: {index_path}")

    index = _load_index(index_path)
    logs_path = packs_dir / "logs.txt"

    manifest = RunManifest.for_sid(sid)
    persist_manifest(
        manifest,
        artifacts={
            "ai_packs": {
                "dir": packs_dir,
                "index": index_path,
                "logs": logs_path,
            }
        },
    )

    total = 0
    successes = 0
    failures = 0
    max_retries = _env_max_retries()
    backoff_schedule = _env_backoff_schedule()
    max_attempts = max(0, max_retries) + 1

    for entry in index:
        if "a" not in entry or "b" not in entry or "file" not in entry:
            raise ValueError(f"Invalid pack index entry: {entry}")
        a_idx = int(entry["a"])
        b_idx = int(entry["b"])
        pack_path = packs_dir / str(entry["file"])
        if not pack_path.exists():
            raise FileNotFoundError(f"Pack file missing: {pack_path}")

        pack = _load_pack(pack_path)
        total += 1

        log = _log_factory(logs_path, sid, {"a": a_idx, "b": b_idx}, pack_path.name)
        log(
            "PACK_START",
            {
                "max_attempts": max_attempts,
            },
        )
        attempts = 0
        decision_payload: Mapping[str, object] | None = None
        last_error: Exception | None = None

        while attempts < max_attempts:
            attempts += 1
            log(
                "REQUEST",
                {"attempt": attempts, "max_attempts": max_attempts},
            )
            try:
                decision_payload = decide_merge_or_different(
                    dict(pack), timeout=AI_REQUEST_TIMEOUT
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                last_error = exc
                log(
                    "ERROR",
                    {
                        "attempt": attempts,
                        "error": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
                if attempts >= max_attempts:
                    decision_payload = None
                    break
                next_attempt = attempts + 1
                delay = _backoff_delay(backoff_schedule, attempts)
                log(
                    "RETRY",
                    {
                        "attempt": next_attempt,
                        "max_attempts": max_attempts,
                        "delay": delay,
                    },
                )
                if delay > 0:
                    time.sleep(delay)
            else:
                log(
                    "RESPONSE",
                    {
                        "attempt": attempts,
                        "payload": decision_payload,
                    },
                )
                break

        if decision_payload is None:
            error_name = last_error.__class__.__name__ if last_error else "UnknownError"
            log(
                "PACK_FAILURE",
                {
                    "attempts": attempts,
                    "error": error_name,
                },
            )
            failures += 1
            continue

        decision_raw = decision_payload.get("decision")
        reason_raw = decision_payload.get("reason")
        decision = str(decision_raw).strip() if decision_raw is not None else ""
        reason = str(reason_raw).strip() if reason_raw is not None else ""
        if decision not in {"merge", "different", "same_debt"} or not reason:
            log(
                "ERROR",
                {
                    "attempt": attempts,
                    "error": "InvalidDecision",
                    "message": "Decision payload missing required fields",
                },
            )
            log(
                "PACK_FAILURE",
                {
                    "attempts": attempts,
                    "error": "InvalidDecision",
                },
            )
            failures += 1
            continue
        timestamp = _isoformat_timestamp()

        a_int = _ensure_int(a_idx, "a_idx")
        b_int = _ensure_int(b_idx, "b_idx")
        payload_for_tags = dict(decision_payload)
        payload_for_tags["decision"] = decision
        payload_for_tags["reason"] = reason

        _write_decision_tags(
            runs_root,
            sid,
            a_int,
            b_int,
            decision,
            reason,
            timestamp,
            payload_for_tags,
        )
        log(
            "PACK_SUCCESS",
            {
                "attempts": attempts,
                "decision": decision,
                "reason": reason,
            },
        )
        successes += 1

    print(
        "[AI] adjudicated {total} packs ({successes} success, {failures} errors)".format(
            total=total, successes=successes, failures=failures
        )
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

