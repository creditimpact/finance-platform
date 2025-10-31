"""Send merge V2 AI packs to the adjudicator service."""

from __future__ import annotations

import argparse
import hashlib
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
    ALLOWED_DECISIONS,
    AdjudicatorError,
    REQUEST_PARAMS as AI_REQUEST_PARAMS,
    RESPONSE_FORMAT as AI_RESPONSE_FORMAT,
    SYSTEM_PROMPT_SHA256,
    _normalize_and_validate_decision,
    decide_merge_or_different,
)
from backend.core.ai.merge_validation import (
    ACCOUNT_STEM_GUARDRAIL_REASON,
    apply_same_account_guardrail,
)
from backend.core.ai.paths import (
    MergePaths,
    ensure_merge_paths,
    merge_paths_from_any,
    pair_pack_filename,
    pair_pack_path,
    pair_result_filename,
    pair_result_path,
    probe_legacy_ai_packs,
)
from backend.core.ai.validators import validate_ai_result
from backend.core.io.tags import read_tags, upsert_tag
from backend.core.logic.report_analysis.account_merge import (
    build_summary_ai_entries,
    build_summary_merge_entry,
    merge_summary_sections,
)
from backend.core.logic.summary_compact import compact_merge_sections
from backend.core.merge.acctnum import normalize_level
from backend.core.runflow import runflow_step
from backend.runflow.decider import finalize_merge_stage
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

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SCHEDULE = "1,3,7"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pack_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


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


def _write_result(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
    path.write_text(f"{serialized}\n", encoding="utf-8")


def _load_result(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, MappingABC) else None


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
    seen_pairs: set[tuple[int, int]] = set()
    for (a_idx, b_idx), entry in sorted_pairs:
        score_value = _score_from_entry(entry)
        pack_entry = dict(entry)
        pack_entry["a"] = a_idx
        pack_entry["b"] = b_idx
        pack_entry["pair"] = [a_idx, b_idx]
        canonical_a, canonical_b = sorted((a_idx, b_idx))
        pack_filename = pair_pack_filename(canonical_a, canonical_b)
        result_filename = pair_result_filename(canonical_a, canonical_b)
        pack_entry["pack_file"] = pack_filename
        pack_entry["result_file"] = result_filename
        pack_entry["score_total"] = score_value
        pack_entry["score"] = score_value
        packs_payload.append(pack_entry)

        pack_file = pack_entry.get("pack_file")
        result_file = pack_entry.get("result_file")
        for pair in ((a_idx, b_idx), (b_idx, a_idx)):
            if pair in seen_pairs:
                continue
            pair_entry: dict[str, object] = {"pair": [pair[0], pair[1]], "score": score_value}
            if pack_file:
                pair_entry["pack_file"] = pack_file
            if result_file:
                pair_entry["result_file"] = result_file
            ai_result = entry.get("ai_result")
            if isinstance(ai_result, MappingABC):
                pair_entry["ai_result"] = dict(ai_result)
            error_payload = entry.get("error")
            if isinstance(error_payload, MappingABC):
                pair_entry["error"] = dict(error_payload)
            pairs_payload.append(pair_entry)
            seen_pairs.add(pair)

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


def _maybe_force_collection_agency_same_debt(
    pack: Mapping[str, object], payload: Mapping[str, object]
) -> tuple[dict[str, object], str | None]:
    """Normalize CA↔CA outcomes toward ``same_debt`` decisions when warranted."""

    context_flags = pack.get("context_flags")
    if not isinstance(context_flags, MappingABC):
        return dict(payload), None

    is_ca_a = bool(context_flags.get("is_collection_agency_a"))
    is_ca_b = bool(context_flags.get("is_collection_agency_b"))
    amounts_equal = bool(context_flags.get("amounts_equal_within_tol"))
    if not (is_ca_a and is_ca_b and amounts_equal):
        return dict(payload), None

    dates_plausible = bool(context_flags.get("dates_plausible_chain"))
    target_decision = "same_debt_diff_account" if dates_plausible else "same_debt_account_unknown"

    current_decision = str(payload.get("decision", "")).strip().lower()
    if current_decision == target_decision:
        return dict(payload), None

    normalized_payload = dict(payload)
    reason = str(normalized_payload.get("reason", "")).strip()
    suffix = (
        "Collection agencies with aligned balances; forced same_debt_diff_account"
        if dates_plausible
        else "Collection agencies with aligned balances; forced same_debt_account_unknown"
    )
    normalized_payload["decision"] = target_decision
    normalized_payload["flags"] = {
        "account_match": False if dates_plausible else "unknown",
        "debt_match": True,
    }
    normalized_payload["normalized"] = True
    normalized_payload["reason"] = f"{reason}; {suffix}" if reason else suffix

    return normalized_payload, target_decision


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
    data = dict(payload)
    if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
        compact_merge_sections(data)
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
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


def _serialize_match_flag(value: bool | str | object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false", "unknown"}:
            return lowered
    return "unknown"


_PAIR_TAG_BY_DECISION: dict[str, str] = {
    "same_account_same_debt": "same_account_pair",
    "same_account_diff_debt": "same_account_pair",
    "same_account_debt_unknown": "same_account_pair",
    "same_debt_diff_account": "same_debt_pair",
    "same_debt_account_unknown": "same_debt_pair",
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
    flags_payload: dict[str, str] | None = None
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

        merge_best_tag: Mapping[str, object] | None = None
        merge_pair_tag: Mapping[str, object] | None = None
        tags_payload = read_tags(tag_path)
        for entry in tags_payload:
            kind = str(entry.get("kind", ""))
            partner_idx = entry.get("with")
            try:
                partner_val = int(partner_idx) if partner_idx is not None else None
            except (TypeError, ValueError):
                partner_val = None
            if partner_val != other_idx:
                continue
            if kind == "merge_best" and merge_best_tag is None:
                merge_best_tag = entry
            elif kind == "merge_pair" and merge_pair_tag is None:
                merge_pair_tag = entry

        summary_path = tag_path.parent / "summary.json"
        summary_payload = _load_summary(summary_path)
        if not isinstance(summary_payload, dict):
            summary_payload = {}

        merge_entries: list[Mapping[str, object]] = []
        if merge_best_tag is not None and any(
            key in merge_best_tag for key in ("parts", "aux", "conflicts", "reasons")
        ):
            merge_entry = build_summary_merge_entry(
                merge_best_tag.get("kind", "merge_best"),
                other_idx,
                merge_best_tag,
                extra={
                    "tiebreaker": merge_best_tag.get("tiebreaker"),
                    "strong_rank": merge_best_tag.get("strong_rank"),
                    "score_total": merge_best_tag.get("score_total"),
                },
            )
            if merge_entry is not None:
                merge_entries.append(merge_entry)
        if merge_pair_tag is not None and any(
            key in merge_pair_tag for key in ("parts", "aux", "conflicts", "reasons")
        ):
            merge_pair_entry = build_summary_merge_entry(
                merge_pair_tag.get("kind", "merge_pair"),
                other_idx,
                merge_pair_tag,
            )
            if merge_pair_entry is not None:
                merge_entries.append(merge_pair_entry)

        ai_entries = build_summary_ai_entries(other_idx, decision, reason, flags_payload)
        summary_changed = merge_summary_sections(
            summary_payload,
            merge_entries=merge_entries,
            ai_entries=ai_entries,
        )

        existing_merge_entries = summary_payload.get("merge_explanations", [])
        has_pair_entry = any(
            isinstance(entry, Mapping)
            and entry.get("kind") == "merge_pair"
            and entry.get("with") == other_idx
            for entry in existing_merge_entries
        )
        if not has_pair_entry:
            pair_source: Mapping[str, object] | None = None
            if merge_pair_tag is not None:
                pair_source = merge_pair_tag
            elif merge_best_tag is not None and any(
                key in merge_best_tag for key in ("parts", "aux")
            ):
                pair_source = merge_best_tag
            else:
                for entry in existing_merge_entries:
                    if (
                        isinstance(entry, Mapping)
                        and entry.get("kind") == "merge_best"
                        and entry.get("with") == other_idx
                    ):
                        pair_source = entry
                        break
            if pair_source is not None:
                pair_entry = build_summary_merge_entry("merge_pair", other_idx, pair_source)
                if pair_entry is not None:
                    if merge_summary_sections(
                        summary_payload, merge_entries=[pair_entry]
                    ):
                        summary_changed = True

        if merge_best_tag is not None:
            points_mode_active = bool(merge_best_tag.get("points_mode"))
            score_total = (
                merge_best_tag.get("score_total")
                or merge_best_tag.get("total")
                or 0
            )
            try:
                score_total_value = (
                    float(score_total)
                    if points_mode_active
                    else int(score_total)
                )
            except (TypeError, ValueError):
                score_total_value = 0.0 if points_mode_active else 0
            reasons_raw = merge_best_tag.get("reasons") or []
            if isinstance(reasons_raw, Iterable) and not isinstance(reasons_raw, (str, bytes)):
                reasons = [str(item) for item in reasons_raw if item is not None]
            else:
                reasons = []
            conflicts_raw = merge_best_tag.get("conflicts") or []
            if isinstance(conflicts_raw, Iterable) and not isinstance(conflicts_raw, (str, bytes)):
                conflicts = [str(item) for item in conflicts_raw if item is not None]
            else:
                conflicts = []
            parts_payload = merge_best_tag.get("parts")
            if isinstance(parts_payload, MappingABC):
                identity_score = _sum_parts(parts_payload, _IDENTITY_PART_FIELDS)
                debt_score = _sum_parts(parts_payload, _DEBT_PART_FIELDS)
            else:
                identity_score = 0
                debt_score = 0
            extras: list[str] = []
            if bool(merge_best_tag.get("strong")):
                extras.append("strong")
            mid_value = merge_best_tag.get("mid")
            try:
                mid_score = (
                    float(mid_value)
                    if points_mode_active
                    else int(mid_value)
                )
            except (TypeError, ValueError):
                mid_score = 0.0 if points_mode_active else 0
            if mid_score > 0:
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
            aux_payload = merge_best_tag.get("aux")
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
                "score_total": score_total_value,
                "reasons": reasons,
                "matched_fields": matched_fields,
                "conflicts": conflicts,
                "identity_score": identity_score,
                "debt_score": debt_score,
                "points_mode": points_mode_active,
            }
            merge_summary["acctnum_level"] = acctnum_level
            summary_payload["merge_scoring"] = merge_summary
            summary_changed = True

        if summary_changed:
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

    merge_paths = ensure_merge_paths(runs_root_path, sid, create=True)
    base_dir = merge_paths.base
    packs_dir = merge_paths.packs_dir
    results_dir = merge_paths.results_dir
    index_path = merge_paths.index_file
    logs_path = merge_paths.log_file

    pack_candidates: list[Path] = []

    if args.packs_dir:
        override_dir = Path(args.packs_dir)
        pack_candidates.append(override_dir)
        log.info("SENDER_PACKS_DIR_OVERRIDE sid=%s dir=%s", sid, override_dir)
    else:
        preferred_dir = manifest.get_ai_packs_dir()
        if preferred_dir is not None:
            preferred_path = Path(preferred_dir)
            pack_candidates.append(preferred_path)
            log.info(
                "SENDER_PACKS_DIR_FROM_MANIFEST sid=%s dir=%s", sid, preferred_path
            )
        manifest_paths = manifest.get_ai_merge_paths()
        legacy_candidate = manifest_paths.get("legacy_packs_dir") if isinstance(manifest_paths, dict) else None
        if isinstance(legacy_candidate, Path):
            pack_candidates.append(legacy_candidate)
            log.info(
                "SENDER_PACKS_DIR_FROM_MANIFEST_LEGACY sid=%s dir=%s",
                sid,
                legacy_candidate,
            )
    pack_candidates.append(packs_dir)

    read_paths: MergePaths | None = None
    for candidate in pack_candidates:
        try:
            candidate_paths = merge_paths_from_any(candidate, create=False)
        except ValueError:
            continue

        candidate_dir = candidate_paths.packs_dir
        if candidate_dir.exists() and any(candidate_dir.glob("pair_*.jsonl")):
            read_paths = candidate_paths
            break

    pack_dir = read_paths.packs_dir if read_paths is not None else None

    legacy_pack_dir = None
    if pack_dir is None:
        legacy_pack_dir = probe_legacy_ai_packs(runs_root_path, sid)
        if legacy_pack_dir is not None:
            pack_dir = legacy_pack_dir
            log.info("SENDER_PACKS_DIR_LEGACY sid=%s dir=%s", sid, legacy_pack_dir)

    if pack_dir is None:
        raise SystemExit(
            "No AI packs for SID (no pair_*.jsonl files found in new or legacy layout)"
        )

    if read_paths is None:
        read_paths = merge_paths

    pair_paths = sorted(pack_dir.glob("pair_*.jsonl"))
    if not pair_paths:
        raise SystemExit(f"No AI packs for SID (no pair_*.jsonl files at {pack_dir})")

    if legacy_pack_dir is not None:
        index_read_path = legacy_pack_dir / "index.json"
    else:
        index_read_path = (
            read_paths.index_file if read_paths.index_file.exists() else None
        )
        if index_read_path is None:
            candidate_index_paths = [pack_dir.parent / "index.json", pack_dir / "index.json"]
            for candidate in candidate_index_paths:
                if candidate.exists():
                    index_read_path = candidate
                    break

    if index_read_path is None or not index_read_path.exists():
        raise SystemExit(
            f"No AI packs for SID (index.json not found for base {base_dir})"
        )

    base_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    run_dir = manifest.path.parent

    index_data, index_entries = _load_index_payload(index_read_path)

    manifest.update_ai_packs(
        dir=base_dir,
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
        pack_sha256 = _pack_sha256(pack_path)
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

        debug_params = dict(AI_REQUEST_PARAMS)
        debug_params["response_format"] = dict(AI_RESPONSE_FORMAT)
        debug_payload = {
            "pack_sha256": pack_sha256,
            "prompt_sha256": SYSTEM_PROMPT_SHA256,
            "model": os.getenv("AI_MODEL"),
            "params": debug_params,
        }

        log.info(
            "AI_DEBUG_METADATA sid=%s a=%s b=%s pack_sha=%s prompt_sha=%s model=%s",
            sid,
            a_idx,
            b_idx,
            pack_sha256,
            SYSTEM_PROMPT_SHA256,
            debug_payload["model"] or "<unset>",
        )

        try:
            a_int = _ensure_int(a_idx, "a_idx")
            b_int = _ensure_int(b_idx, "b_idx")
        except (TypeError, ValueError):
            a_int = b_int = None

        result_payload_existing: Mapping[str, object] | None = None
        legacy_result_found = False
        if a_int is not None and b_int is not None:
            result_path = pair_result_path(merge_paths, a_int, b_int)
            result_payload_existing = _load_result(result_path)

        if result_payload_existing is None:
            ai_result_existing = pack.get("ai_result")
            if isinstance(ai_result_existing, MappingABC):
                result_payload_existing = dict(ai_result_existing)
                legacy_result_found = True

        if result_payload_existing is not None:
            entry["ai_result"] = dict(result_payload_existing)
            entry.pop("error", None)
            if legacy_result_found and a_int is not None and b_int is not None:
                result_path = pair_result_path(merge_paths, a_int, b_int)
                _write_result(result_path, result_payload_existing)
            pack_log(
                "PACK_SKIP",
                {"reason": "result_present", "debug": debug_payload},
            )
            successes += 1
            continue

        attempts = 0
        decision_payload: Mapping[str, object] | None = None
        last_error: Exception | None = None

        pack_log(
            "PACK_START",
            {
                "max_attempts": max_attempts,
                "debug": debug_payload,
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
                    "debug": debug_payload,
                },
            )
            timestamp = _isoformat_timestamp()
            _write_error_tags(run_dir, a_idx, b_idx, error_name, message, timestamp)
            entry["error"] = {"kind": error_name, "message": message}
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
                    "debug": debug_payload,
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
            entry["error"] = {"kind": "InvalidDecision", "message": str(exc)}
            entry.pop("ai_result", None)
            failures += 1
            continue

        validation_ok, validation_error = validate_ai_result(normalized_payload)
        fallback_used = False
        if not validation_ok:
            fallback_used = True
            log.warning(
                "AI_DECISION_INVALID sid=%s a=%s b=%s reason=%s",
                sid,
                a_idx,
                b_idx,
                validation_error or "unknown",
            )
            pack_log(
                "AI_DECISION_INVALID",
                {
                    "error": validation_error or "unknown",
                },
            )
            normalized_payload = {
                "decision": "different",
                "reason": "invalid_response_schema",
                "flags": {"account_match": False, "debt_match": False},
            }

        forced_decision = None
        if not fallback_used:
            normalized_payload, forced_decision = _maybe_force_collection_agency_same_debt(
                pack, normalized_payload
            )
            if forced_decision is not None:
                was_normalized = True
                forced_ok, forced_error = validate_ai_result(normalized_payload)
                if not forced_ok:
                    fallback_used = True
                    log.warning(
                        "AI_DECISION_FORCED_INVALID sid=%s a=%s b=%s reason=%s",
                        sid,
                        a_idx,
                        b_idx,
                        forced_error or "unknown",
                    )
                    pack_log(
                        "CA_DECISION_OVERRIDE_INVALID",
                        {"error": forced_error or "unknown"},
                    )
                    normalized_payload = {
                        "decision": "different",
                        "reason": "invalid_response_schema",
                        "flags": {"account_match": False, "debt_match": False},
                    }
                else:
                    log.info(
                        "AI_DECISION_CA_OVERRIDE sid=%s a=%s b=%s forced=%s",
                        sid,
                        a_idx,
                        b_idx,
                        forced_decision,
                    )
                    pack_log(
                        "CA_DECISION_OVERRIDE",
                        {"forced_decision": forced_decision},
                    )

        guardrail_reason: str | None = None
        if not fallback_used:
            normalized_payload, guardrail_reason = apply_same_account_guardrail(
                pack, normalized_payload
            )
            if guardrail_reason is not None:
                was_normalized = True
                guardrail_ok, guardrail_error = validate_ai_result(normalized_payload)
                if not guardrail_ok:
                    fallback_used = True
                    log.warning(
                        "AI_DECISION_GUARDRAIL_INVALID sid=%s a=%s b=%s reason=%s",
                        sid,
                        a_idx,
                        b_idx,
                        guardrail_error or "unknown",
                    )
                    pack_log(
                        "GUARDRAIL_OVERRIDE_INVALID",
                        {"error": guardrail_error or "unknown"},
                    )
                    normalized_payload = {
                        "decision": "different",
                        "reason": "invalid_response_schema",
                        "flags": {"account_match": False, "debt_match": False},
                    }
                else:
                    log.info(
                        "AI_DECISION_GUARDRAIL sid=%s a=%s b=%s forced=different reason=%s",
                        sid,
                        a_idx,
                        b_idx,
                        guardrail_reason,
                    )
                    metrics_reason = (
                        "stems_conflict"
                        if guardrail_reason == ACCOUNT_STEM_GUARDRAIL_REASON
                        else guardrail_reason
                    )
                    runflow_step(
                        sid,
                        "merge",
                        "acct_guardrail",
                        account=str(a_idx),
                        metrics={"reason": metrics_reason},
                        out={"pair": f"{a_idx}-{b_idx}"},
                    )
                    runflow_step(
                        sid,
                        "merge",
                        "acct_guardrail",
                        account=str(b_idx),
                        metrics={"reason": metrics_reason},
                        out={"pair": f"{a_idx}-{b_idx}"},
                    )
                    pack_log(
                        "GUARDRAIL_OVERRIDE",
                        {
                            "forced_decision": "different",
                            "reason": guardrail_reason,
                        },
                    )

        original_decision_raw = decision_payload.get("decision")
        original_decision = (
            str(original_decision_raw).strip().lower()
            if original_decision_raw is not None
            else ""
        )
        decision = normalized_payload["decision"]
        reason = normalized_payload["reason"]
        flags_normalized = normalized_payload["flags"]
        flags_serialized = {
            "account_match": _serialize_match_flag(
                flags_normalized.get("account_match")
            ),
            "debt_match": _serialize_match_flag(flags_normalized.get("debt_match")),
        }
        flags_for_storage = {
            "account_match": flags_normalized.get("account_match"),
            "debt_match": flags_normalized.get("debt_match"),
        }

        if decision not in ALLOWED_DECISIONS:
            raise AdjudicatorError(
                f"Decision outside contract: {decision!r}"
            )

        if was_normalized and not fallback_used:
            log.info(
                "AI_DECISION_NORMALIZED sid=%s a=%s b=%s from=%s to=%s",
                sid,
                a_idx,
                b_idx,
                original_decision or "<missing>",
                decision,
            )

        timestamp = _isoformat_timestamp()

        if a_int is None or b_int is None:
            a_int = _ensure_int(a_idx, "a_idx")
            b_int = _ensure_int(b_idx, "b_idx")

        payload_for_tags = dict(normalized_payload)
        payload_for_tags["flags"] = dict(flags_serialized)

        log.info(
            "AI_DECISION_FINAL sid=%s a=%s b=%s decision=%s flags=%s",
            sid,
            a_idx,
            b_idx,
            decision,
            json.dumps(flags_serialized, sort_keys=True),
        )

        pack_log(
            "AI_DECISION_PARSED",
            {
                "decision": decision,
                "flags": flags_serialized,
                "normalized": was_normalized and not fallback_used,
                "fallback": fallback_used,
            },
        )

        ai_result_payload = {
            "decision": decision,
            "reason": reason,
            "flags": dict(flags_for_storage),
            "debug": dict(debug_payload),
        }
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
                "fallback": fallback_used,
                "debug": debug_payload,
            },
        )
        result_path = pair_result_path(merge_paths, a_int, b_int)
        _write_result(result_path, ai_result_payload)
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

        try:
            finalize_merge_stage(sid, runs_root=runs_root_path)
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "RUNFLOW_MERGE_STAGE_FINALIZE_FAILED sid=%s", sid, exc_info=True
            )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

