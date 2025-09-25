"""Canonical tag compaction helpers for AI adjudication artifacts."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Union

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

Pathish = Union[str, Path, PathLike[str]]


def _coerce_int(value: object) -> int | None:
    try:
        if isinstance(value, bool):
            return int(value)
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _sum_parts(parts: Mapping[str, object] | None, fields: Iterable[str]) -> int:
    total = 0
    if isinstance(parts, Mapping):
        for field in fields:
            part_value = parts.get(field)
            coerced = _coerce_int(part_value)
            if coerced is not None:
                total += coerced
    return total


def _normalize_matched_fields(raw: Mapping[str, object] | None) -> dict[str, bool]:
    matched: dict[str, bool] = {}
    if not isinstance(raw, Mapping):
        return matched
    for field, flag in raw.items():
        matched[str(field)] = bool(flag)
    return matched


def _build_merge_scoring_summary(
    best_tag: Mapping[str, object] | None,
    existing: Mapping[str, object] | None,
) -> dict[str, object] | None:
    summary: dict[str, object] = {}
    if isinstance(existing, Mapping):
        summary.update(existing)

    if not isinstance(best_tag, Mapping):
        return summary or None

    parts = best_tag.get("parts") if isinstance(best_tag.get("parts"), Mapping) else None
    identity_score = _sum_parts(parts, _IDENTITY_PART_FIELDS)
    debt_score = _sum_parts(parts, _DEBT_PART_FIELDS)

    aux_payload = best_tag.get("aux") if isinstance(best_tag.get("aux"), Mapping) else {}
    acctnum_level = "none"
    if isinstance(aux_payload, Mapping):
        acctnum_level = str(aux_payload.get("acctnum_level") or "none")
        if acctnum_level == "none":
            account_number_aux = aux_payload.get("account_number")
            if isinstance(account_number_aux, Mapping):
                acctnum_level = str(account_number_aux.get("acctnum_level") or "none")
        matched_fields = _normalize_matched_fields(aux_payload.get("matched_fields"))
    else:
        matched_fields = {}

    conflicts_raw = best_tag.get("conflicts")
    conflicts: list[str] = []
    if isinstance(conflicts_raw, Iterable) and not isinstance(conflicts_raw, (str, bytes)):
        for conflict in conflicts_raw:
            if conflict is None:
                continue
            conflict_text = str(conflict)
            if conflict_text:
                conflicts.append(conflict_text)

    reasons_raw = best_tag.get("reasons")
    reasons: list[str] = []
    if isinstance(reasons_raw, Iterable) and not isinstance(reasons_raw, (str, bytes)):
        for reason in reasons_raw:
            if reason is None:
                continue
            reason_text = str(reason)
            if reason_text:
                reasons.append(reason_text)

    partner = _coerce_int(best_tag.get("with"))
    score_total = _coerce_int(best_tag.get("score_total"))
    if score_total is None:
        score_total = _coerce_int(best_tag.get("total"))
    if score_total is None:
        score_total = 0

    summary.update(
        {
            "identity_score": identity_score,
            "debt_score": debt_score,
            "acctnum_level": acctnum_level,
            "matched_fields": matched_fields,
            "conflicts": conflicts,
            "score_total": score_total,
        }
    )

    if reasons:
        summary["reasons"] = reasons
    elif "reasons" in summary:
        summary.pop("reasons")

    if partner is not None:
        summary["best_with"] = partner
    elif "best_with" in summary:
        summary.pop("best_with")

    return summary


def _minimal_issue(tag: Mapping[str, object]) -> dict[str, object]:
    payload = {"kind": "issue"}
    type_value = _coerce_str(tag.get("type"))
    if type_value:
        payload["type"] = type_value
    return payload


def _minimal_merge_best(tag: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {"kind": "merge_best"}
    partner = _coerce_int(tag.get("with"))
    if partner is not None:
        payload["with"] = partner
    decision = _coerce_str(tag.get("decision"))
    if decision:
        payload["decision"] = decision
    timestamp = _coerce_str(tag.get("at"))
    if timestamp:
        payload["at"] = timestamp
    return payload


def _normalize_flag(value: object) -> bool | str | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "unknown":
            return "unknown"
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _minimal_ai_decision(tag: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {"kind": "ai_decision"}
    partner = _coerce_int(tag.get("with"))
    if partner is not None:
        payload["with"] = partner
    decision = _coerce_str(tag.get("decision"))
    if decision:
        payload["decision"] = decision
    timestamp = _coerce_str(tag.get("at"))
    if timestamp:
        payload["at"] = timestamp
    flags = tag.get("flags")
    if isinstance(flags, Mapping):
        account_flag = _normalize_flag(flags.get("account_match"))
        debt_flag = _normalize_flag(flags.get("debt_match"))
        filtered_flags = {}
        if account_flag is not None:
            filtered_flags["account_match"] = account_flag
        if debt_flag is not None:
            filtered_flags["debt_match"] = debt_flag
        if filtered_flags:
            payload["flags"] = filtered_flags
    return payload


def _minimal_pair(tag: Mapping[str, object], *, kind: str) -> dict[str, object]:
    payload: dict[str, object] = {"kind": kind}
    partner = _coerce_int(tag.get("with"))
    if partner is not None:
        payload["with"] = partner
    timestamp = _coerce_str(tag.get("at"))
    if timestamp:
        payload["at"] = timestamp
    return payload


def _merge_explanation_from_tag(tag: Mapping[str, object]) -> dict[str, object] | None:
    kind = str(tag.get("kind", ""))
    if kind not in {"merge_best", "merge_pair"}:
        return None

    payload: dict[str, object] = {"kind": kind}
    partner = _coerce_int(tag.get("with"))
    if partner is not None:
        payload["with"] = partner
    decision = _coerce_str(tag.get("decision"))
    if decision:
        payload["decision"] = decision

    verbose_fields: dict[str, object | None] = {
        "total": tag.get("total"),
        "mid": tag.get("mid"),
        "dates_all": tag.get("dates_all"),
        "parts": tag.get("parts"),
        "aux": tag.get("aux"),
        "reasons": tag.get("reasons"),
        "conflicts": tag.get("conflicts"),
        "strong": tag.get("strong"),
        "strong_rank": tag.get("strong_rank"),
        "score_total": tag.get("score_total"),
        "tiebreaker": tag.get("tiebreaker"),
    }

    meaningful = False
    for key, value in verbose_fields.items():
        if _has_value(value):
            payload[key] = value
            meaningful = True

    aux = tag.get("aux")
    if isinstance(aux, Mapping):
        acct_level = aux.get("acctnum_level")
        if _has_value(acct_level):
            payload.setdefault("acctnum_level", acct_level)
            meaningful = True
        matched_fields = aux.get("matched_fields")
        if isinstance(matched_fields, Mapping) and matched_fields:
            payload.setdefault("matched_fields", dict(matched_fields))
            meaningful = True

    return payload if meaningful else None


def _ai_explanations_from_tag(
    tag: Mapping[str, object],
    *,
    decision_reason_map: dict[int, str],
) -> list[dict[str, object]]:
    kind = str(tag.get("kind", ""))
    partner = _coerce_int(tag.get("with"))
    decision = _coerce_str(tag.get("decision"))
    reason = tag.get("reason")
    raw_response = tag.get("raw_response")
    entries: list[dict[str, object]] = []

    if kind == "ai_decision":
        if isinstance(partner, int) and isinstance(reason, str) and reason:
            decision_reason_map[partner] = reason
        if not _has_value(reason) and not _has_value(raw_response):
            return entries
        payload: dict[str, object] = {"kind": kind}
        if partner is not None:
            payload["with"] = partner
        if decision:
            payload["decision"] = decision
        if _has_value(reason):
            payload["reason"] = reason
        flags = tag.get("flags")
        if isinstance(flags, Mapping):
            filtered_flags: dict[str, object] = {}
            account_flag = _normalize_flag(flags.get("account_match"))
            debt_flag = _normalize_flag(flags.get("debt_match"))
            if account_flag is not None:
                filtered_flags["account_match"] = account_flag
            if debt_flag is not None:
                filtered_flags["debt_match"] = debt_flag
            if filtered_flags:
                payload["flags"] = filtered_flags
        if _has_value(raw_response):
            payload["raw_response"] = raw_response
        entries.append(payload)
        return entries

    if kind in {"same_debt_pair", "same_account_pair"}:
        if not _has_value(reason) and isinstance(partner, int):
            reason = decision_reason_map.get(partner)
        if not _has_value(reason):
            return entries
        payload = {"kind": kind}
        if partner is not None:
            payload["with"] = partner
        payload["reason"] = reason
        entries.append(payload)

    return entries


def _load_tags(tags_path: Path) -> list[Mapping[str, object]]:
    if not tags_path.exists():
        return []
    try:
        raw = tags_path.read_text(encoding="utf-8")
    except OSError:
        return []
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if isinstance(data, list):
        entries = data
    elif isinstance(data, Mapping):
        tags = data.get("tags")
        entries = tags if isinstance(tags, list) else []
    else:
        entries = []
    result: list[Mapping[str, object]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            result.append(entry)
    return result


def _dedupe(entries: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    unique: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in entries:
        key = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _merge_summary_entries(
    existing: Sequence[MutableMapping[str, object]] | None,
    updates: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    combined: list[dict[str, object]] = []
    if existing:
        for entry in existing:
            if isinstance(entry, Mapping):
                combined.append(dict(entry))
    for entry in updates:
        if isinstance(entry, Mapping):
            combined.append(dict(entry))
    return _dedupe(combined)


def _write_json(path: Path, payload: object) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(f"{serialized}\n", encoding="utf-8")


def compact_account_tags(
    account_dir: Pathish,
    *,
    minimal_only: bool = True,
    explanations_to_summary: bool = True,
) -> None:
    """Reduce ``tags.json`` to minimal tags and move verbose data to summary."""

    account_path = Path(account_dir)
    tags_path = account_path / "tags.json"
    tags = _load_tags(tags_path)
    if not tags:
        return

    minimal_tags: list[dict[str, object]] = []
    merge_explanations: list[dict[str, object]] = []
    ai_explanations: list[dict[str, object]] = []
    decision_reasons: dict[int, str] = {}
    best_merge_tag: Mapping[str, object] | None = None

    for tag in tags:
        kind = str(tag.get("kind", "")).strip().lower()
        if minimal_only:
            if kind == "issue":
                minimal_tags.append(_minimal_issue(tag))
            elif kind == "merge_best":
                minimal_tags.append(_minimal_merge_best(tag))
            elif kind == "ai_decision":
                minimal_tags.append(_minimal_ai_decision(tag))
            elif kind in {"same_debt_pair", "same_account_pair"}:
                minimal_tags.append(_minimal_pair(tag, kind=kind))
        else:
            minimal_tags.append(dict(tag))

        if explanations_to_summary:
            if kind in {"merge_best", "merge_pair"}:
                merge_payload = _merge_explanation_from_tag(tag)
                if merge_payload is not None:
                    merge_explanations.append(merge_payload)
                if kind == "merge_best":
                    best_merge_tag = dict(tag)
            elif kind in {"ai_decision", "same_debt_pair", "same_account_pair"}:
                ai_explanations.extend(
                    _ai_explanations_from_tag(tag, decision_reason_map=decision_reasons)
                )
        elif kind == "merge_best":
            best_merge_tag = dict(tag)

    if minimal_only:
        minimal_tags = [tag for tag in minimal_tags if tag]
        minimal_tags = _dedupe(minimal_tags)

    _write_json(tags_path, minimal_tags)

    if not explanations_to_summary:
        return

    summary_path = account_path / "summary.json"
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary_data = {}
    else:
        summary_data = {}
    if not isinstance(summary_data, dict):
        summary_data = {}

    existing_merge = summary_data.get("merge_explanations")
    existing_ai = summary_data.get("ai_explanations")
    existing_scoring = summary_data.get("merge_scoring")

    summary_data["merge_explanations"] = _merge_summary_entries(existing_merge, merge_explanations)
    summary_data["ai_explanations"] = _merge_summary_entries(existing_ai, ai_explanations)

    merge_scoring_summary = _build_merge_scoring_summary(best_merge_tag, existing_scoring)
    if merge_scoring_summary is not None:
        summary_data["merge_scoring"] = merge_scoring_summary
    elif "merge_scoring" in summary_data:
        summary_data.pop("merge_scoring", None)

    _write_json(summary_path, summary_data)


def compact_tags_for_run(
    sid: str,
    *,
    runs_root: Pathish | None = None,
    minimal_only: bool = True,
    explanations_to_summary: bool = True,
) -> None:
    """Compact tags for all accounts under ``runs/<sid>/cases/accounts``."""

    base = Path(runs_root) if runs_root is not None else Path("runs")
    accounts_dir = base / sid / "cases" / "accounts"
    if not accounts_dir.exists() or not accounts_dir.is_dir():
        return

    for account_dir in sorted(accounts_dir.iterdir()):
        if account_dir.is_dir():
            compact_account_tags(
                account_dir,
                minimal_only=minimal_only,
                explanations_to_summary=explanations_to_summary,
            )


def compact_tags_for_sid(
    sid: str,
    runs_root: Pathish | None = None,
    *,
    minimal_only: bool = True,
    explanations_to_summary: bool = True,
) -> None:
    """Backwards-compatible alias for run-level compaction."""

    compact_tags_for_run(
        sid,
        runs_root=runs_root,
        minimal_only=minimal_only,
        explanations_to_summary=explanations_to_summary,
    )


__all__ = [
    "compact_account_tags",
    "compact_tags_for_run",
    "compact_tags_for_sid",
]
