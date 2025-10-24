"""Utilities for building note_style AI packs with contextual metadata."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.core.ai.paths import (
    NoteStyleAccountPaths,
    ensure_note_style_account_paths,
    ensure_note_style_paths,
)


_NOTE_VALUE_PATHS: tuple[tuple[str, ...], ...] = (
    ("note",),
    ("note_text",),
    ("explain",),
    ("explanation",),
    ("data", "explain"),
    ("answers", "explain"),
    ("answers", "explanation"),
    ("answers", "note"),
    ("answers", "notes"),
    ("answers", "customer_note"),
)

_BUREAU_FIELDS: tuple[str, ...] = (
    "reported_creditor",
    "account_type",
    "account_status",
    "payment_status",
    "creditor_type",
    "date_opened",
    "date_reported",
    "date_of_last_activity",
    "closed_date",
    "last_verified",
    "balance_owed",
    "high_balance",
    "past_due_amount",
)

_AMOUNT_FIELDS = {"balance_owed", "high_balance", "past_due_amount"}
_DATE_FIELDS = {
    "date_opened",
    "date_reported",
    "date_of_last_activity",
    "closed_date",
    "last_verified",
}

_BUREAU_PRIORITY = ("transunion", "experian", "equifax")

_SYSTEM_MESSAGE = (
    "You analyze customer notes and return a concise style extract. "
    "Focus on tone, contextual hints, and emphasis. "
    "Use account_context and bureaus_summary as background; do not restate them verbatim. "
    "Respond with valid JSON that includes tone, context, and emphasis details."
)


class PackBuilderError(RuntimeError):
    """Raised when a note_style pack cannot be constructed."""


def build_pack(
    sid: str,
    account_id: str,
    *,
    runs_root: Path | str | None = None,
    mirror_debug: bool = True,
) -> Mapping[str, Any]:
    """Build a note_style pack for ``sid``/``account_id``.

    The pack is persisted to the canonical packs directory as a single JSONL line.
    The constructed payload is returned for convenience.
    """

    if not sid:
        raise ValueError("sid is required")
    if not account_id:
        raise ValueError("account_id is required")

    runs_root_path = Path(runs_root or "runs").resolve()
    run_dir = runs_root_path / sid

    response_path = run_dir / "frontend" / "review" / "responses" / f"{account_id}.result.json"
    if not response_path.is_file():
        raise PackBuilderError(f"response note not found: {response_path}")

    account_dir = _locate_account_dir(run_dir / "cases" / "accounts", account_id)
    if account_dir is None:
        raise PackBuilderError(
            f"account artifacts not found for account_id={account_id!r} under {run_dir / 'cases' / 'accounts'}"
        )

    response_payload = _load_json(response_path)
    note_text = _extract_note_text(response_payload)

    bureaus_payload = _ensure_mapping(_load_json(account_dir / "bureaus.json"))
    tags_payload = _ensure_sequence(_load_json(account_dir / "tags.json"))
    meta_payload = _ensure_mapping(_load_json(account_dir / "meta.json"))

    bureaus_summary = _summarize_bureaus(bureaus_payload)
    account_context = _build_account_context(meta_payload, bureaus_payload, tags_payload, bureaus_summary)

    pack_payload = {
        "sid": sid,
        "account_id": account_id,
        "channel": "frontend_review",
        "note_text": note_text,
        "account_context": account_context,
        "bureaus_summary": bureaus_summary,
        "messages": [
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": {"note_text": note_text}},
        ],
    }

    paths = ensure_note_style_paths(runs_root_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    _write_jsonl(account_paths.pack_file, pack_payload)

    if mirror_debug:
        _write_debug_snapshot(account_paths, pack_payload)

    return pack_payload


def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def _write_debug_snapshot(account_paths: NoteStyleAccountPaths, payload: Mapping[str, Any]) -> None:
    account_paths.debug_file.parent.mkdir(parents=True, exist_ok=True)
    account_paths.debug_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    return json.loads(text)


def _ensure_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _ensure_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _extract_note_text(payload: Any) -> str:
    if isinstance(payload, Mapping):
        for path in _NOTE_VALUE_PATHS:
            current: Any = payload
            for key in path:
                if not isinstance(current, Mapping):
                    break
                current = current.get(key)
            else:
                normalized = _normalize_text(current)
                if normalized:
                    return normalized
    return ""


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return unicodedata.normalize("NFKC", value).strip()
    return unicodedata.normalize("NFKC", str(value)).strip()


def _normalize_amount(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        decimal_value = Decimal(str(value))
    else:
        text = _normalize_text(value)
        if not text:
            return ""
        stripped = text.replace(",", "").replace("$", "").strip()
        negative = False
        if stripped.startswith("(") and stripped.endswith(")"):
            negative = True
            stripped = stripped[1:-1]
        if stripped.startswith("-"):
            negative = True
            stripped = stripped[1:]
        if stripped.endswith("-"):
            negative = True
            stripped = stripped[:-1]
        stripped = stripped.strip()
        if not stripped:
            return ""
        try:
            decimal_value = Decimal(stripped)
        except InvalidOperation:
            filtered = re.sub(r"[^0-9.]", "", stripped)
            if not filtered:
                return ""
            try:
                decimal_value = Decimal(filtered)
            except InvalidOperation:
                return ""
        if negative:
            decimal_value = -decimal_value
    normalized = format(decimal_value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if not normalized:
        normalized = "0"
    return normalized


def _normalize_date(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    stripped = text.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y"):
        try:
            parsed = datetime.strptime(stripped, fmt)
        except ValueError:
            continue
        return parsed.date().isoformat()
    for fmt in ("%m/%d/%y", "%m-%d-%y", "%m.%d.%y"):
        try:
            parsed = datetime.strptime(stripped, fmt)
        except ValueError:
            continue
        return parsed.date().isoformat()
    try:
        parsed = datetime.fromisoformat(stripped)
    except ValueError:
        return stripped
    return parsed.date().isoformat()


def _normalize_field(field: str, value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("value", "raw", "display", "text", "formatted"):
            if key in value:
                candidate = _normalize_field(field, value[key])
                if candidate:
                    return candidate
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for entry in value:
            candidate = _normalize_field(field, entry)
            if candidate:
                return candidate
        return ""
    if field in _DATE_FIELDS:
        return _normalize_date(value)
    if field in _AMOUNT_FIELDS:
        return _normalize_amount(value)
    return _normalize_text(value)


def _summarize_bureaus(bureaus: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(bureaus, Mapping):
        return {}

    per_bureau: dict[str, dict[str, str]] = {}
    field_values: dict[str, dict[str, str]] = {field: {} for field in _BUREAU_FIELDS}

    for bureau_name, payload in sorted(bureaus.items(), key=lambda item: item[0]):
        if not isinstance(payload, Mapping):
            continue
        normalized_fields: dict[str, str] = {}
        for field in _BUREAU_FIELDS:
            value = _normalize_field(field, payload.get(field))
            if value:
                normalized_fields[field] = value
                field_values.setdefault(field, {})[bureau_name] = value
        if normalized_fields:
            per_bureau[bureau_name] = normalized_fields

    if not per_bureau:
        return {}

    majority_values: dict[str, str] = {}
    disagreements: dict[str, dict[str, str]] = {}

    for field, bureau_map in field_values.items():
        if not bureau_map:
            continue
        unique_values = {value for value in bureau_map.values() if value}
        if unique_values:
            selected = _select_majority_value(bureau_map)
            if selected:
                majority_values[field] = selected
        if len(unique_values) > 1:
            disagreements[field] = dict(sorted(bureau_map.items(), key=lambda item: item[0]))

    return {
        "per_bureau": per_bureau,
        "majority_values": majority_values,
        "disagreements": disagreements,
    }


def _select_majority_value(bureau_map: Mapping[str, str]) -> str:
    non_empty = {bureau: value for bureau, value in bureau_map.items() if value}
    if not non_empty:
        return ""

    counter = Counter(non_empty.values())
    if counter:
        most_common = counter.most_common()
        if most_common:
            top_count = most_common[0][1]
            candidates = [value for value, count in most_common if count == top_count]
            if len(candidates) == 1:
                return candidates[0]
            for bureau in _BUREAU_PRIORITY:
                candidate = non_empty.get(bureau)
                if candidate:
                    return candidate
            for bureau, value in sorted(non_empty.items(), key=lambda item: item[0]):
                if value:
                    return value
    for bureau in _BUREAU_PRIORITY:
        candidate = non_empty.get(bureau)
        if candidate:
            return candidate
    for value in non_empty.values():
        if value:
            return value
    return ""


def _build_account_context(
    meta: Mapping[str, Any],
    bureaus: Mapping[str, Any],
    tags: Sequence[Any],
    bureaus_summary: Mapping[str, Any],
) -> dict[str, Any]:
    context: dict[str, Any] = {}

    heading_guess = _normalize_text(meta.get("heading_guess"))
    creditor_name = _normalize_text(meta.get("creditor_name"))
    reported_creditor = heading_guess or creditor_name

    per_bureau = bureaus_summary.get("per_bureau") if isinstance(bureaus_summary, Mapping) else None
    if not reported_creditor:
        majority_values = bureaus_summary.get("majority_values") if isinstance(bureaus_summary, Mapping) else None
        if isinstance(majority_values, Mapping):
            reported_creditor = _normalize_text(majority_values.get("reported_creditor"))
        if not reported_creditor and isinstance(per_bureau, Mapping):
            for bureau in _BUREAU_PRIORITY:
                payload = per_bureau.get(bureau)
                if isinstance(payload, Mapping):
                    candidate = _normalize_text(payload.get("reported_creditor"))
                    if candidate:
                        reported_creditor = candidate
                        break
    if reported_creditor:
        context["reported_creditor"] = reported_creditor

    account_tail = _extract_account_tail(meta, bureaus)
    if account_tail:
        context["account_tail"] = account_tail

    issues: list[str] = []
    for tag in tags:
        if not isinstance(tag, Mapping):
            continue
        if _normalize_text(tag.get("kind")).lower() != "issue":
            continue
        issue_value = _normalize_text(tag.get("type"))
        if issue_value and issue_value not in issues:
            issues.append(issue_value)
    if issues:
        context.setdefault("tags", {})["issues"] = issues
        context["primary_issue"] = issues[0]

    if heading_guess:
        context.setdefault("meta", {})["heading_guess"] = heading_guess

    return context


def _extract_account_tail(meta: Mapping[str, Any], bureaus: Mapping[str, Any]) -> str:
    tail = _normalize_text(meta.get("account_number_tail"))
    if tail:
        digits = re.sub(r"\D", "", tail)
        return digits[-4:] if digits else tail

    if isinstance(bureaus, Mapping):
        for payload in bureaus.values():
            if not isinstance(payload, Mapping):
                continue
            candidate = _normalize_text(payload.get("account_number_display"))
            if candidate:
                digits = re.sub(r"\D", "", candidate)
                return digits[-4:] if digits else candidate
    return ""


def _locate_account_dir(accounts_dir: Path, account_id: str) -> Path | None:
    if not accounts_dir.is_dir():
        return None

    # direct match
    direct = accounts_dir / account_id
    if direct.is_dir():
        return direct.resolve()

    digits = re.findall(r"(\d+)", account_id)
    for piece in digits:
        normalized = piece.lstrip("0") or "0"
        candidate = accounts_dir / normalized
        if candidate.is_dir():
            return candidate.resolve()

    target = _normalize_text(account_id).lower()
    for entry in sorted(accounts_dir.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        for filename in ("summary.json", "meta.json"):
            payload = _load_json(entry / filename)
            if isinstance(payload, Mapping):
                for key in ("account_id", "account_key", "account_identifier"):
                    value = _normalize_text(payload.get(key)).lower()
                    if value and value == target:
                        return entry.resolve()
    return None


__all__ = ["build_pack", "PackBuilderError"]
