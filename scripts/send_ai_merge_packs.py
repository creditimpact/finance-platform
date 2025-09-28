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
from backend.core.ai.adjudicator import (
    AdjudicatorError,
    _normalize_and_validate_decision,
    decide_merge_or_different,
)
from backend.core.io.tags import read_tags, upsert_tag
from backend.core.merge.acctnum import normalize_level
from backend.pipeline.runs import RunManifest, persist_manifest

log = logging.getLogger(__name__)



_DEBT_RULES_TEXT = (
    "Debt rules:\n"
    "- If Balance Owed and Past Due are both zero on both sides, this indicates no active debt; do NOT treat this as \"same debt\".\n"
    '- Do not use "0 == 0" as evidence of the same obligation.\n'
    '- Only consider "same_debt" (or variants) when there is a positive amount or explicit textual corroboration.\n'
    '- If both balances and past-due are zero and identifiers do not strongly match, prefer "different".'
)
_DEBT_RULES_MARKER = (
    "- If Balance Owed and Past Due are both zero on both sides, this indicates no active debt; do NOT treat this as \"same debt\"."
)


def _append_zero_debt_rules(pack: Mapping[str, object]) -> dict[str, object]:
    """Ensure the system prompt contains explicit zero-debt guidance."""

    def _list_contains_marker(items: Iterable[object]) -> bool:
        for item in items:
            if isinstance(item, str) and _DEBT_RULES_MARKER in item:
                return True
        return False

    updated_pack = dict(pack)

    messages = updated_pack.get("messages")
    if isinstance(messages, list):
        new_messages: list[object] = []
        system_found = False
        for message in messages:
            if isinstance(message, MappingABC):
                msg_copy = dict(message)
                role = str(msg_copy.get("role", "")).strip().lower()
                if role == "system":
                    system_found = True
                    content = msg_copy.get("content")
                    if isinstance(content, str):
                        if _DEBT_RULES_MARKER not in content:
                            trimmed = content.rstrip()
                            if trimmed:
                                trimmed += "\n"
                            msg_copy["content"] = f"{trimmed}{_DEBT_RULES_TEXT}"
                    elif isinstance(content, list):
                        if not _list_contains_marker(content):
                            msg_copy["content"] = [*content, _DEBT_RULES_TEXT]
                    elif isinstance(content, MappingABC):
                        content_copy = dict(content)
                        rules = content_copy.get("rules")
                        if isinstance(rules, list):
                            if not _list_contains_marker(rules):
                                content_copy["rules"] = [*rules, _DEBT_RULES_TEXT]
                        else:
                            content_copy["rules"] = [_DEBT_RULES_TEXT]
                        msg_copy["content"] = content_copy
                new_messages.append(msg_copy)
            else:
                new_messages.append(message)

        if not system_found:
            new_messages = [{"role": "system", "content": _DEBT_RULES_TEXT}, *new_messages]
        updated_pack["messages"] = new_messages
        return updated_pack

    system_content = updated_pack.get("system")
    if isinstance(system_content, str):
        if _DEBT_RULES_MARKER not in system_content:
            trimmed = system_content.rstrip()
            if trimmed:
                trimmed += "\n"
            updated_pack["system"] = f"{trimmed}{_DEBT_RULES_TEXT}"
    elif isinstance(system_content, list):
        if not _list_contains_marker(system_content):
            updated_pack["system"] = [*system_content, _DEBT_RULES_TEXT]
    elif isinstance(system_content, MappingABC):
        content_copy = dict(system_content)
        rules = content_copy.get("rules")
        if isinstance(rules, list):
            if not _list_contains_marker(rules):
                content_copy["rules"] = [*rules, _DEBT_RULES_TEXT]
        else:
            content_copy["rules"] = [_DEBT_RULES_TEXT]
        updated_pack["system"] = content_copy
    else:
        updated_pack["system"] = _DEBT_RULES_TEXT

    return updated_pack



def _packs_dir_for(sid: str, runs_root: Path) -> Path:
    """Return the canonical ``ai_packs`` directory for ``sid``."""

    from backend.pipeline.auto_ai import packs_dir_for as _packs_dest  # local import to avoid cycles

    return _packs_dest(sid, runs_root=runs_root)


DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SCHEDULE = "1,3,7"


def _load_index_payload(
    path: Path,
) -> tuple[dict[str, object], dict[tuple[int, int], dict[str, object]]]:
    data_raw: dict[str, object] = {}
    entries: dict[tuple[int, int], dict[str, object]] = {}

    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, MappingABC):
            data_raw = dict(loaded)
        elif isinstance(loaded, list):  # pragma: no cover - legacy support
            data_raw = {"packs": list(loaded)}
        else:
            raise ValueError(f"Pack index must be a mapping or list: {path}")

    for source_key in ("packs", "pairs"):
        items = data_raw.get(source_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, MappingABC):
                continue
            a_val = item.get("a")
            b_val = item.get("b")
            pair_val = item.get("pair")
            a_idx: int | None = None
            b_idx: int | None = None
            if isinstance(pair_val, Sequence) and len(pair_val) == 2:
                try:
                    a_idx = int(pair_val[0])
                    b_idx = int(pair_val[1])
                except (TypeError, ValueError):
                    a_idx = b_idx = None
            if a_idx is None or b_idx is None:
                try:
                    a_idx = int(a_val)
                    b_idx = int(b_val)
                except (TypeError, ValueError):
                    continue
            key = (a_idx, b_idx)
            dest = entries.setdefault(key, {"a": a_idx, "b": b_idx})
            dest.update({k: v for k, v in dict(item).items() if k not in {"a", "b"}})
            dest["pair"] = [a_idx, b_idx]
    return data_raw, entries


def _normalize_pair(value: Mapping[str, object] | Sequence[object] | None) -> tuple[int, int] | None:
    if isinstance(value, MappingABC):
        try:
            return int(value.get("a")), int(value.get("b"))
        except (TypeError, ValueError):
            return None
    if isinstance(value, Sequence) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _pair_from_filename(name: str) -> tuple[int, int] | None:
    if not name.startswith("pair_"):
        return None
    stem = name
    if stem.endswith(".jsonl"):
        stem = stem[:-6]
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    try:
        return int(parts[1]), int(parts[2])
    except (TypeError, ValueError):
        return None


def _pair_from_pack(pack: Mapping[str, object], pack_path: Path) -> tuple[int, int]:
    pair_value = pack.get("pair")
    normalized = _normalize_pair(pair_value if isinstance(pair_value, (MappingABC, Sequence)) else None)
    if normalized is not None:
        return normalized
    filename_pair = _pair_from_filename(pack_path.name)
    if filename_pair is not None:
        return filename_pair
    raise ValueError(f"Pack {pack_path} is missing pair indices")


def _score_from_pack(pack: Mapping[str, object]) -> int:
    highlights = pack.get("highlights")
    total: object | None = None
    if isinstance(highlights, MappingABC):
        total = highlights.get("total")
    elif isinstance(pack.get("score"), (int, float)):
        total = pack["score"]
    try:
        return int(total) if total is not None else 0
    except (TypeError, ValueError):
        return 0


def _write_pack(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
    path.write_text(f"{serialized}\n", encoding="utf-8")


def _score_from_entry(entry: Mapping[str, object]) -> int:
    for key in ("score", "score_total"):
        value = entry.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _build_index_payload(
    sid: str,
    base: Mapping[str, object],
    entries: Mapping[tuple[int, int], Mapping[str, object]],
) -> dict[str, object]:
    sorted_pairs = sorted(entries.items(), key=lambda item: item[0])
    packs_payload: list[dict[str, object]] = []
    pairs_payload: list[dict[str, object]] = []
    for (a_idx, b_idx), entry in sorted_pairs:
        score_value = _score_from_entry(entry)
        pack_entry = dict(entry)
        pack_entry["a"] = a_idx
        pack_entry["b"] = b_idx
        pack_entry["pair"] = [a_idx, b_idx]
        pack_entry.setdefault("pack_file", f"pair_{a_idx:03d}_{b_idx:03d}.jsonl")
        pack_entry["score_total"] = score_value
        pack_entry["score"] = score_value
        packs_payload.append(pack_entry)

        pair_entry: dict[str, object] = {"pair": [a_idx, b_idx], "score": score_value}
        ai_result = entry.get("ai_result")
        if isinstance(ai_result, MappingABC):
            pair_entry["ai_result"] = dict(ai_result)
        error_payload = entry.get("error")
        if isinstance(error_payload, MappingABC):
            pair_entry["error"] = dict(error_payload)
        pairs_payload.append(pair_entry)

    payload = dict(base)
    payload["sid"] = sid
    payload["packs"] = packs_payload
    payload["pairs"] = pairs_payload
    payload["pairs_count"] = len(packs_payload)
    return payload


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


def _serialize_match_flag(value: bool | str | object) -> bool | str:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "unknown":
            return "unknown"
    return "unknown"


_DECISION_CONTRACT = [
    "merge",
    "same_debt",
    "same_debt_account_diff",
    "same_account",
    "same_account_debt_diff",
    "different",
]

_DECISION_CONTRACT_TEXT = (
    '"decision": ["merge", "same_debt", "same_debt_account_diff", '
    '"same_account", "same_account_debt_diff", "different"]'
)


_PAIR_TAG_BY_DECISION: dict[str, str] = {
    "same_account": "same_account_pair",
    "same_account_debt_diff": "same_account_pair",
    "merge": "same_account_pair",
    "same_debt": "same_debt_pair",
    "same_debt_account_diff": "same_debt_pair",
}

_IDENTITY_PART_FIELDS = {
    "account_number",
    "creditor_type",
    "date_opened",
    "closed_date",
    "date_of_last_activity",
    "date_reported",
    "last_verified",
}

_DEBT_PART_FIELDS = {
    "balance_owed",
    "high_balance",
    "past_due_amount",
    "last_payment",
}


def _sum_parts(parts: Mapping[str, object] | None, fields: Iterable[str]) -> int:
    total = 0
    if not isinstance(parts, MappingABC):
        return total
    for field in fields:
        try:
            total += int(parts.get(field, 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


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


def _prune_pair_tags(tag_path: Path, other_idx: int, *, keep_kind: str | None) -> None:
    existing_tags = read_tags(tag_path)
    if not existing_tags:
        return

    filtered: list[dict[str, object]] = []
    modified = False
    for entry in existing_tags:
        kind = str(entry.get("kind", "")).strip().lower()
        if kind not in {"same_account_pair", "same_debt_pair"}:
            filtered.append(dict(entry))
            continue
        source = str(entry.get("source", ""))
        if source != "ai_adjudicator":
            filtered.append(dict(entry))
            continue
        partner_raw = entry.get("with")
        try:
            partner = int(partner_raw) if partner_raw is not None else None
        except (TypeError, ValueError):
            partner = None
        if partner != other_idx:
            filtered.append(dict(entry))
            continue
        if keep_kind is not None and kind == keep_kind:
            filtered.append(dict(entry))
            continue
        modified = True

    if not modified:
        return

    serialized = json.dumps(filtered, ensure_ascii=False, indent=2)
    tag_path.write_text(f"{serialized}\n", encoding="utf-8")


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
    raw_flags = payload.get("flags")
    flags_payload: dict[str, bool | str] | None = None
    if isinstance(raw_flags, MappingABC):
        account_flag = _serialize_match_flag(raw_flags.get("account_match"))
        debt_flag = _serialize_match_flag(raw_flags.get("debt_match"))
        flags_payload = {"account_match": account_flag, "debt_match": debt_flag}
    pair_tag_kind = _PAIR_TAG_BY_DECISION.get(decision)

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
        if flags_payload is not None:
            decision_tag["flags"] = dict(flags_payload)
        raw_response = payload.get("raw_response")
        if raw_response is not None:
            decision_tag["raw_response"] = raw_response
        upsert_tag(tag_path, decision_tag, unique_keys=("kind", "with", "source"))

        if pair_tag_kind is not None:
            pair_tag = {
                "kind": pair_tag_kind,
                "with": other_idx,
                "source": "ai_adjudicator",
                "reason": reason,
                "at": timestamp,
            }
            upsert_tag(tag_path, pair_tag, unique_keys=("kind", "with", "source"))
            _prune_pair_tags(tag_path, other_idx, keep_kind=pair_tag_kind)
        else:
            _prune_pair_tags(tag_path, other_idx, keep_kind=None)

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
            parts_payload = merge_tag.get("parts")
            if isinstance(parts_payload, MappingABC):
                identity_score = _sum_parts(parts_payload, _IDENTITY_PART_FIELDS)
                debt_score = _sum_parts(parts_payload, _DEBT_PART_FIELDS)
            else:
                identity_score = 0
                debt_score = 0
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
            acctnum_level = normalize_level(None)
            if isinstance(aux_payload, MappingABC):
                raw_matched = aux_payload.get("matched_fields")
                if isinstance(raw_matched, MappingABC):
                    matched_fields = {
                        str(field): bool(flag) for field, flag in raw_matched.items()
                    }
                acct_val = aux_payload.get("acctnum_level")
                acctnum_level = normalize_level(acct_val)
            merge_summary: dict[str, object] = {
                "best_with": other_idx,
                "score_total": score_total_int,
                "reasons": reasons,
                "matched_fields": matched_fields,
                "conflicts": conflicts,
                "identity_score": identity_score,
                "debt_score": debt_score,
            }
            merge_summary["acctnum_level"] = acctnum_level
            summary_payload["merge_scoring"] = merge_summary
            _write_summary(summary_path, summary_payload)

    log.info(
        "AI_TAGS_WRITTEN a=%s b=%s decision=%s pair_tag=%s",
        a_idx,
        b_idx,
        decision,
        pair_tag_kind or "none",
    )

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

    pair_paths = sorted(packs_dir.glob("pair_*.jsonl"))
    if not pair_paths:
        raise SystemExit(f"No AI packs for SID (no pair_*.jsonl files at {packs_dir})")

    run_dir = manifest.path.parent

    index_data, index_entries = _load_index_payload(index_path)
    logs_path = packs_dir / "logs.txt"

    manifest.update_ai_packs(
        dir=packs_dir,
        index=index_path,
        logs=logs_path,
        pairs=len(pair_paths),
    )

    total = len(pair_paths)
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

    for pack_path in pair_paths:
        pack = _append_zero_debt_rules(_load_pack(pack_path))
        a_idx, b_idx = _pair_from_pack(pack, pack_path)
        score_value = _score_from_pack(pack)
        entry = index_entries.setdefault((a_idx, b_idx), {"a": a_idx, "b": b_idx})
        entry.update(
            {
                "pack_file": pack_path.name,
                "score": score_value,
                "score_total": score_value,
                "pair": [a_idx, b_idx],
            }
        )

        pair_for_log = {"a": a_idx, "b": b_idx}
        pack_log = _log_factory(logs_path, sid, pair_for_log, pack_path.name)

        ai_result_existing = pack.get("ai_result")
        if isinstance(ai_result_existing, MappingABC):
            entry["ai_result"] = dict(ai_result_existing)
            entry.pop("error", None)
            pack_log("PACK_SKIP", {"reason": "ai_result_present"})
            successes += 1
            continue

        attempts = 0
        decision_payload: Mapping[str, object] | None = None
        last_error: Exception | None = None

        pack_log(
            "PACK_START",
            {
                "max_attempts": max_attempts,
            },
        )

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
            error_payload = {"kind": error_name, "message": message}
            pack["ai_error"] = error_payload
            pack.pop("ai_result", None)
            _write_pack(pack_path, pack)
            entry["error"] = dict(error_payload)
            entry.pop("ai_result", None)
            failures += 1
            continue

        try:
            normalized_payload, was_normalized = _normalize_and_validate_decision(
                decision_payload
            )
        except AdjudicatorError as exc:
            pack_log(
                "ERROR",
                {
                    "attempt": attempts,
                    "error": "InvalidDecision",
                    "message": str(exc),
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
                str(exc),
                timestamp,
            )
            failures += 1
            continue

        original_decision_raw = decision_payload.get("decision")
        original_decision = (
            str(original_decision_raw).strip().lower()
            if original_decision_raw is not None
            else ""
        )
        decision = normalized_payload["decision"]
        reason = normalized_payload["reason"]
        flags_normalized = normalized_payload["flags"]

        if decision not in _DECISION_CONTRACT:
            raise AdjudicatorError(
                f"Decision outside contract: {decision!r}"
            )

        if was_normalized:
            log.info(
                "AI_DECISION_NORMALIZED sid=%s a=%s b=%s from=%s to=%s",
                sid,
                a_idx,
                b_idx,
                original_decision or "<missing>",
                decision,
            )

        timestamp = _isoformat_timestamp()

        a_int = _ensure_int(a_idx, "a_idx")
        b_int = _ensure_int(b_idx, "b_idx")
        payload_for_tags = dict(normalized_payload)

        log.info(
            "AI_DECISION_FINAL sid=%s a=%s b=%s decision=%s flags=%s",
            sid,
            a_idx,
            b_idx,
            decision,
            json.dumps(flags_normalized, sort_keys=True),
        )

        pack_log(
            "AI_DECISION_PARSED",
            {
                "decision": decision,
                "flags": flags_normalized,
                "normalized": was_normalized,
            },
        )

        ai_result_payload = {
            "decision": decision,
            "reason": reason,
            "flags": dict(flags_normalized),
        }
        pack["ai_result"] = ai_result_payload
        pack.pop("ai_error", None)
        _write_pack(pack_path, pack)
        entry["ai_result"] = dict(ai_result_payload)
        entry.pop("error", None)

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

    index_payload = _build_index_payload(sid, index_data, index_entries)
    index_serialized = json.dumps(index_payload, ensure_ascii=False, indent=2)
    index_path.write_text(f"{index_serialized}\n", encoding="utf-8")

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

