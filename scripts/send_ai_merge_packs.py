"""Send merge V2 AI packs to the adjudicator service."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Mapping as MappingABC
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence, overload

try:  # pragma: no cover - convenience bootstrap for direct execution
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback when bootstrap is unavailable
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from backend.config import AI_REQUEST_TIMEOUT
from backend.core.ai.adjudicator import decide_merge_or_different
from backend.core.io.tags import read_tags, upsert_tag
from backend.pipeline.runs import RunManifest, persist_manifest

log = logging.getLogger(__name__)



def _packs_dir_for(sid: str, runs_root: Path) -> Path:
    """Return the canonical ``ai_packs`` directory for ``sid``."""

    from backend.pipeline.auto_ai import packs_dir_for as _packs_dest  # local import to avoid cycles

    return _packs_dest(sid, runs_root=runs_root)


DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SCHEDULE = "1,3,7"


def _load_index(path: Path) -> list[Mapping[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, MappingABC):
        entries = data.get("pairs")
        if entries is None:
            entries = data.get("packs")
        if entries is None:
            return []
        if not isinstance(entries, list):
            raise ValueError(f"Pack index entries must be a list: {path}")
        return [dict(entry) for entry in entries if isinstance(entry, MappingABC)]
    if isinstance(data, list):  # pragma: no cover - legacy support
        return [dict(entry) for entry in data if isinstance(entry, MappingABC)]
    raise ValueError(f"Pack index must be a mapping or list: {path}")


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


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _load_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_summary(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(dict(payload), ensure_ascii=False, indent=2)
    path.write_text(f"{serialized}\n", encoding="utf-8")


def _log_factory(path: Path, sid: str, pair: Mapping[str, int], pack_file: str):
    def _log(event: str, payload: Mapping[str, object] | None = None) -> None:
        extras: dict[str, object] = {
            "sid": sid,
            "pair": {"a": pair["a"], "b": pair["b"]},
            "pack_file": pack_file,
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


def _accounts_root(run_dir: Path) -> Path:
    return run_dir / "cases" / "accounts"


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


@overload
def _write_decision_tags(
    run_path: Path | str,
    a_idx: int,
    b_idx: int,
    decision: str,
    reason: str,
    timestamp: str,
    payload: Mapping[str, object],
) -> None:
    ...


@overload
def _write_decision_tags(
    run_path: Path | str,
    sid: str,
    a_idx: int,
    b_idx: int,
    decision: str,
    reason: str,
    timestamp: str,
    payload: Mapping[str, object],
) -> None:
    ...


def _write_decision_tags(run_path: Path | str, *args: object) -> None:
    if len(args) == 6:
        a_idx, b_idx, decision, reason, timestamp, payload = args
        run_dir = Path(run_path)
    elif len(args) == 7:
        sid, a_idx, b_idx, decision, reason, timestamp, payload = args
        run_dir = Path(run_path) / str(sid)
    else:  # pragma: no cover - defensive
        raise TypeError("_write_decision_tags expects 7 or 8 arguments")

    account_a = _ensure_int(a_idx, "a_idx")
    account_b = _ensure_int(b_idx, "b_idx")

    if not isinstance(payload, MappingABC):
        raise TypeError("payload must be a mapping")

    _write_decision_tags_resolved(
        run_dir,
        account_a,
        account_b,
        str(decision),
        str(reason),
        str(timestamp),
        payload,
    )


def _write_decision_tags_resolved(
    run_dir: Path,
    a_idx: int,
    b_idx: int,
    decision: str,
    reason: str,
    timestamp: str,
    payload: Mapping[str, object],
) -> None:
    base = _accounts_root(run_dir)
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

        merge_tag: Mapping[str, object] | None = None
        for entry in read_tags(tag_path):
            if str(entry.get("kind", "")) != "merge_best":
                continue
            partner_idx = entry.get("with")
            try:
                partner_val = int(partner_idx) if partner_idx is not None else None
            except (TypeError, ValueError):
                partner_val = None
            if partner_val == other_idx:
                merge_tag = entry
                break

        if merge_tag is not None:
            summary_path = tag_path.parent / "summary.json"
            summary_payload = _load_summary(summary_path)
            if not isinstance(summary_payload, dict):
                summary_payload = {}
            score_total = merge_tag.get("score_total") or merge_tag.get("total") or 0
            try:
                score_total_int = int(score_total)
            except (TypeError, ValueError):
                score_total_int = 0
            reasons_raw = merge_tag.get("reasons") or []
            if isinstance(reasons_raw, Iterable) and not isinstance(reasons_raw, (str, bytes)):
                reasons = [str(item) for item in reasons_raw if item is not None]
            else:
                reasons = []
            conflicts_raw = merge_tag.get("conflicts") or []
            if isinstance(conflicts_raw, Iterable) and not isinstance(conflicts_raw, (str, bytes)):
                conflicts = [str(item) for item in conflicts_raw if item is not None]
            else:
                conflicts = []
            extras: list[str] = []
            if bool(merge_tag.get("strong")):
                extras.append("strong")
            mid_value = merge_tag.get("mid")
            try:
                mid_int = int(mid_value)
            except (TypeError, ValueError):
                mid_int = 0
            if mid_int > 0:
                extras.append("mid")
            extras.append("total")
            unique_reasons: list[str] = []
            for item in reasons + extras:
                if not isinstance(item, str):
                    continue
                if item not in unique_reasons:
                    unique_reasons.append(item)
            reasons = unique_reasons
            matched_fields: dict[str, bool] = {}
            aux_payload = merge_tag.get("aux")
            acctnum_level: str | None = None
            if isinstance(aux_payload, MappingABC):
                raw_matched = aux_payload.get("matched_fields")
                if isinstance(raw_matched, MappingABC):
                    matched_fields = {
                        str(field): bool(flag) for field, flag in raw_matched.items()
                    }
                acct_val = aux_payload.get("acctnum_level")
                if isinstance(acct_val, str) and acct_val:
                    acctnum_level = acct_val
            merge_summary: dict[str, object] = {
                "best_with": other_idx,
                "score_total": score_total_int,
                "reasons": reasons,
                "matched_fields": matched_fields,
                "conflicts": conflicts,
            }
            if acctnum_level:
                merge_summary["acctnum_level"] = acctnum_level
            summary_payload["merge_scoring"] = merge_summary
            _write_summary(summary_path, summary_payload)

def _write_error_tags(
    run_dir: Path,
    a_idx: int,
    b_idx: int,
    error_kind: str,
    message: str,
    timestamp: str,
) -> None:
    base = _accounts_root(run_dir)
    for source_idx, other_idx in ((a_idx, b_idx), (b_idx, a_idx)):
        tag_path = base / str(source_idx) / "tags.json"
        error_tag = {
            "kind": "ai_error",
            "with": other_idx,
            "error_kind": error_kind,
            "message": message,
            "source": "ai_adjudicator",
            "at": timestamp,
        }
        upsert_tag(tag_path, error_tag, unique_keys=("kind", "with", "source"))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True, help="Session identifier")
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Root directory containing runs/<SID> outputs",
    )
    parser.add_argument(
        "--packs-dir",
        default=None,
        help="Optional override for the directory containing ai packs",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum number of retries before failing (defaults to AI_MAX_RETRIES)",
    )
    parser.add_argument(
        "--backoff",
        default=None,
        help="Comma-separated backoff schedule in seconds (defaults to AI_BACKOFF_SCHEDULE)",
    )
    args = parser.parse_args(argv)

    sid = str(args.sid)

    if args.runs_root:
        runs_root_path = Path(args.runs_root)
    else:
        runs_root_path = Path(os.environ.get("RUNS_ROOT", "runs"))

    manifest = RunManifest.for_sid(sid)

    if args.packs_dir:
        packs_dir = Path(args.packs_dir)
        log.info("SENDER_PACKS_DIR_OVERRIDE sid=%s dir=%s", sid, packs_dir)
    else:
        preferred_dir = manifest.get_ai_packs_dir()
        if preferred_dir is not None:
            packs_dir = Path(preferred_dir)
            log.info("SENDER_PACKS_DIR_FROM_MANIFEST sid=%s dir=%s", sid, packs_dir)
        else:
            packs_dir = _packs_dir_for(sid, runs_root_path)
            log.info("SENDER_PACKS_DIR_FALLBACK sid=%s dir=%s", sid, packs_dir)

    if not packs_dir.exists():
        raise SystemExit(f"No AI packs for SID (directory not found at {packs_dir})")

    index_path = packs_dir / "index.json"
    if not index_path.exists():
        raise SystemExit(f"No AI packs for SID (index.json not found at {index_path})")

    run_dir = manifest.path.parent

    index = _load_index(index_path)
    logs_path = packs_dir / "logs.txt"

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
    if args.max_retries is None:
        max_retries = _env_max_retries()
    else:
        max_retries = max(0, int(args.max_retries))

    if args.backoff is None:
        backoff_schedule = _env_backoff_schedule()
    else:
        backoff_schedule = _parse_backoff_schedule(str(args.backoff))
    max_attempts = max(0, max_retries) + 1

    for entry in index:
        if "a" not in entry or "b" not in entry or "pack_file" not in entry:
            raise ValueError(f"Invalid pack index entry: {entry}")
        a_idx = int(entry["a"])
        b_idx = int(entry["b"])
        pack_path = packs_dir / str(entry["pack_file"])
        if not pack_path.exists():
            raise FileNotFoundError(f"Pack file missing: {pack_path}")

        pack = _load_pack(pack_path)
        total += 1

        pack_log = _log_factory(logs_path, sid, {"a": a_idx, "b": b_idx}, pack_path.name)
        pack_log(
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
            pack_log(
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
                pack_log(
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
                pack_log(
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
                pack_log(
                    "RESPONSE",
                    {
                        "attempt": attempts,
                        "payload": decision_payload,
                    },
                )
                break

        if decision_payload is None:
            error_name = last_error.__class__.__name__ if last_error else "UnknownError"
            message = str(last_error) if last_error else ""
            pack_log(
                "PACK_FAILURE",
                {
                    "attempts": attempts,
                    "error": error_name,
                },
            )
            timestamp = _isoformat_timestamp()
            _write_error_tags(run_dir, a_idx, b_idx, error_name, message, timestamp)
            failures += 1
            continue

        decision_raw = decision_payload.get("decision")
        reason_raw = decision_payload.get("reason")
        decision_value = str(decision_raw).strip().lower() if decision_raw is not None else ""
        reason = str(reason_raw).strip() if reason_raw is not None else ""
        if decision_value not in {"merge", "different", "same_debt"} or not reason:
            pack_log(
                "ERROR",
                {
                    "attempt": attempts,
                    "error": "InvalidDecision",
                    "message": "Decision payload missing required fields",
                },
            )
            pack_log(
                "PACK_FAILURE",
                {
                    "attempts": attempts,
                    "error": "InvalidDecision",
                },
            )
            timestamp = _isoformat_timestamp()
            _write_error_tags(
                run_dir,
                a_idx,
                b_idx,
                "InvalidDecision",
                "Decision payload missing required fields",
                timestamp,
            )
            failures += 1
            continue
        decision = decision_value
        timestamp = _isoformat_timestamp()

        a_int = _ensure_int(a_idx, "a_idx")
        b_int = _ensure_int(b_idx, "b_idx")
        payload_for_tags = dict(decision_payload)
        payload_for_tags["decision"] = decision
        payload_for_tags["reason"] = reason

        _write_decision_tags(
            run_dir,
            a_int,
            b_int,
            decision,
            reason,
            timestamp,
            payload_for_tags,
        )
        pack_log(
            "PACK_SUCCESS",
            {
                "attempts": attempts,
                "decision": decision,
                "reason": reason,
            },
        )
        successes += 1

    if failures == 0:
        manifest.set_ai_sent()
        persist_manifest(manifest)
        log.info("MANIFEST_AI_SENT sid=%s", sid)

    print(
        "[AI] adjudicated {total} packs ({successes} success, {failures} errors)".format(
            total=total, successes=successes, failures=failures
        )
    )

    if failures == 0:
        try:
            from backend.core.logic.tags.compact import compact_tags_for_sid

            compact_tags_for_sid(sid)
        except Exception as exc:  # pragma: no cover - defensive logging
            log.warning("TAGS_COMPACT_SKIP sid=%s err=%s", sid, exc)
        else:
            manifest.set_ai_compacted()
            persist_manifest(manifest)
            log.info("MANIFEST_AI_COMPACTED sid=%s", sid)
            log.info("TAGS_COMPACTED sid=%s", sid)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

