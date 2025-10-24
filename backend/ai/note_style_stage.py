"""Build note_style AI stage artifacts from frontend review responses."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import secrets
import threading
import time
import uuid
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

try:  # pragma: no cover - platform dependent
    import fcntl  # type: ignore[import]
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore[assignment]

from backend import config
from backend.ai.manifest import ensure_note_style_section, register_note_style_build
from backend.ai.note_style_logging import append_note_style_warning, log_structured_event
from backend.core.ai.paths import (
    NoteStyleAccountPaths,
    NoteStylePaths,
    ensure_note_style_account_paths,
    ensure_note_style_paths,
    note_style_pack_filename,
    note_style_result_filename,
    normalize_note_style_account_id,
)
from backend.pipeline.runs import RunManifest
from backend.runflow.decider import record_stage


log = logging.getLogger(__name__)

_INDEX_SCHEMA_VERSION = 1
_DEBOUNCE_MS_ENV = "NOTE_STYLE_DEBOUNCE_MS"
_DEFAULT_DEBOUNCE_MS = 750

_NOTE_STYLE_MODEL = config.NOTE_STYLE_MODEL
_NOTE_TEXT_MAX_CHARS = config.NOTE_STYLE_MAX_NOTE_CHARS
_NOTE_STYLE_SYSTEM_PROMPT = (
    "You extract structured style from a customer's free-text note.\n"
    "Respond in JSON ONLY using EXACT schema: {{\"tone\": <string>, \"context_hints\": {{\"timeframe\": {{\"month\": <string|null>, \"relative\": <string|null>}}, \"topic\": <string>, \"entities\": {{\"creditor\": <string|null>, \"amount\": <number|null>}}}}, \"emphasis\": [<string>...], \"confidence\": <float>, \"risk_flags\": [<string>...]}}. No prose or markdown.\n"
    "Rules:\n"
    "- Base output only on note_text; treat all other fields as context hints, not facts to restate.\n"
    "- Keep every value short; lists must contain no more than 6 items.\n"
    "- Summarize to tags/short phrases; do not copy sentences verbatim.\n"
    "- If note is empty/meaningless → tone=\"neutral\", topic=\"unspecified\", confidence<=0.2, add risk_flags [\"empty_note\"].\n"
    "- If note asserts a legal claim but mentions no supporting docs → add [\"unsupported_claim\"].\n"
    "- Calibrate confidence: short/ambiguous notes ≤0.5.\n"
    "Examples:\n"
    "  * note_text=\"\" → risk_flags [\"empty_note\"].\n"
    "  * note_text=\"They owe me $500 but I have no documents\" → add [\"unsupported_claim\"].\n"
    "Prompt salt: {prompt_salt}\n"
)

_ALLOWED_TONES = {
    "neutral",
    "calm",
    "confident",
    "assertive",
    "empathetic",
    "formal",
    "conversational",
    "factual",
}

_ALLOWED_TOPICS = {
    "payment_dispute",
    "not_mine",
    "billing_error",
    "identity_theft",
    "late_fee",
    "other",
}

_ALLOWED_EMPHASIS = {
    "paid_already",
    "inaccurate_reporting",
    "identity_concerns",
    "support_request",
    "fee_waiver",
    "ownership_dispute",
    "update_requested",
    "evidence_provided",
}

_RELATIVE_TIMEFRAME_PATTERNS: dict[str, tuple[str, ...]] = {
    "last_two_months": (
        r"last\s+two\s+months",
        r"past\s+two\s+months",
        r"last\s+couple\s+of\s+months",
        r"last\s+couple\s+months",
    ),
    "last_month": (r"last\s+month", r"previous\s+month", r"past\s+month"),
    "current_month": (r"this\s+month", r"current\s+month"),
    "next_month": (r"next\s+month",),
    "last_year": (
        r"last\s+year",
        r"previous\s+year",
        r"past\s+year",
        r"past\s+twelve\s+months",
    ),
}

_MONTH_NAME_MAP = {
    "jan": "Jan",
    "january": "Jan",
    "feb": "Feb",
    "february": "Feb",
    "mar": "Mar",
    "march": "Mar",
    "apr": "Apr",
    "april": "Apr",
    "may": "May",
    "jun": "Jun",
    "june": "Jun",
    "jul": "Jul",
    "july": "Jul",
    "aug": "Aug",
    "august": "Aug",
    "sep": "Sep",
    "sept": "Sep",
    "september": "Sep",
    "oct": "Oct",
    "october": "Oct",
    "nov": "Nov",
    "november": "Nov",
    "dec": "Dec",
    "december": "Dec",
}

_KNOWN_CREDITORS = {
    "capital one": "Capital One",
    "bank of america": "Bank of America",
    "wells fargo": "Wells Fargo",
    "chase": "Chase",
    "discover": "Discover",
    "synchrony": "Synchrony",
    "citibank": "Citibank",
    "navy federal": "Navy Federal",
}

_AMOUNT_PATTERN = re.compile(r"\$?\b\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\b")
_CREDITOR_PATTERN = re.compile(
    r"\b(?:with|from|at|to|by|for)\s+([A-Z][\w&]*(?:\s+[A-Z][\w&]*){0,3})"
)
_MONTH_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(_MONTH_NAME_MAP.keys(), key=len, reverse=True)) + r")\b(?:[-/,\s]*(\d{2,4}))?",
    re.IGNORECASE,
)
_PERSONAL_DATA_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{3}[-\.\s]\d{3}[-\.\s]\d{4}\b"),
    re.compile(r"\b\d{9}\b"),
    re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),
)

_EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
_PHONE_PATTERNS = (
    re.compile(r"(?<!\d)(?:\+?1[-\.\s]*)?(?:\(\d{3}\)|\d{3})[-\.\s]*\d{3}[-\.\s]*\d{4}(?!\d)"),
    re.compile(r"\b\d{3}[-\.\s]\d{3}[-\.\s]\d{4}\b"),
)

_NOTE_VALUE_PATHS: tuple[tuple[str, ...], ...] = (
    ("data", "explain"),
    ("answers", "explain"),
    ("answers", "explanation"),
    ("note",),
    ("explain",),
    ("explanation",),
    ("answers", "note"),
    ("answers", "notes"),
    ("answers", "customer_note"),
)

_ZERO_WIDTH_WHITESPACE = {
    ord("\u200b"): " ",  # zero width space
    ord("\u200c"): " ",  # zero width non-joiner
    ord("\u200d"): " ",  # zero width joiner
    ord("\ufeff"): " ",  # byte order mark / zero width no-break space
    ord("\u2060"): " ",  # word joiner
}


_LOW_SIGNAL_PHRASES = {
    "help",
    "please help",
    "need help",
    "please fix",
    "fix please",
    "fix this",
    "fix it",
    "please assist",
    "assist please",
    "need assistance",
}


_INDEX_LOCK_POLL_INTERVAL = 0.05
_INDEX_LOCK_STALE_TIMEOUT = 30.0


@dataclass(frozen=True)
class _LoadedResponseNote:
    account_id: str
    note_raw: str
    note_sanitized: str
    source_path: Path
    source_hash: str
    ui_allegations_selected: tuple[str, ...] = ()


class NoteStyleSkip(Exception):
    """Raised when the note_style stage should soft-skip processing."""

    def __init__(self, reason: str, *, detail: str | None = None) -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        return Path(env_value) if env_value else Path("runs")
    return Path(runs_root)


@dataclass(frozen=True)
class NoteStyleResponseAccount:
    """Metadata for a single response discovered for the note_style stage."""

    account_id: str
    normalized_account_id: str
    response_path: Path
    response_relative: PurePosixPath
    pack_filename: str
    result_filename: str
    account_dir: Path | None = None


@dataclass(frozen=True)
class _NoteStyleSourcePaths:
    manifest: Mapping[str, Any] | None
    accounts_dir: Path
    responses_dir: Path


_ACCOUNT_MAP_CACHE: dict[tuple[str, str], tuple[float, dict[str, Path]]] = {}
_ACCOUNT_MAP_LOCK = threading.Lock()


def _manifest_path_from_value(raw: Any, run_dir: Path) -> Path:
    candidate = Path(str(raw))
    if not candidate.is_absolute():
        candidate = (run_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _load_manifest_payload(run_dir: Path) -> Mapping[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    payload = _load_json_mapping(manifest_path)
    return payload if isinstance(payload, Mapping) else None


def _resolve_source_paths(sid: str, runs_root: Path) -> _NoteStyleSourcePaths:
    run_dir = (runs_root / sid).resolve()
    manifest_payload = _load_manifest_payload(run_dir)

    accounts_dir = run_dir / "cases" / "accounts"
    responses_dir = run_dir / "frontend" / "review" / "responses"

    if isinstance(manifest_payload, Mapping):
        base_dirs = manifest_payload.get("base_dirs")
        if isinstance(base_dirs, Mapping):
            accounts_raw = base_dirs.get("cases_accounts_dir")
            if accounts_raw:
                try:
                    accounts_dir = _manifest_path_from_value(accounts_raw, run_dir)
                except OSError:
                    pass

        frontend_section = manifest_payload.get("frontend")
        if isinstance(frontend_section, Mapping):
            responses_raw = (
                frontend_section.get("responses_dir")
                or frontend_section.get("results_dir")
                or frontend_section.get("responses")
                or frontend_section.get("results")
            )
            if responses_raw:
                try:
                    responses_dir = _manifest_path_from_value(responses_raw, run_dir)
                except OSError:
                    pass

    return _NoteStyleSourcePaths(
        manifest=manifest_payload,
        accounts_dir=accounts_dir,
        responses_dir=responses_dir,
    )


def _canonical_response_reference(
    runs_root: Path, response_path: Path
) -> PurePosixPath:
    resolved_root = runs_root.resolve()
    resolved_path = response_path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError:
        return PurePosixPath(resolved_path.as_posix())
    relative_posix = PurePosixPath(relative.as_posix())
    return PurePosixPath("runs") / relative_posix


def _register_account_identifier(
    mapping: dict[str, Path], identifier: str, entry: Path
) -> None:
    if not identifier:
        return
    if identifier not in mapping:
        mapping[identifier] = entry
    normalized = normalize_note_style_account_id(identifier)
    if normalized and normalized not in mapping:
        mapping[normalized] = entry


def _build_account_directory_map(accounts_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if not accounts_dir.is_dir():
        return mapping

    for entry in sorted(accounts_dir.iterdir(), key=lambda path: path.name):
        if not entry.is_dir():
            continue

        resolved_entry = entry.resolve()
        mapping.setdefault(entry.name, resolved_entry)

        summary = _load_json_mapping(resolved_entry / "summary.json")
        identifiers: list[str] = []
        if isinstance(summary, Mapping):
            for key in ("account_id", "account_key", "account_identifier"):
                value = _normalize_text(summary.get(key))
                if value:
                    identifiers.append(value)

        if not identifiers:
            meta_payload = _load_json_mapping(resolved_entry / "meta.json")
            if isinstance(meta_payload, Mapping):
                candidate = _normalize_text(meta_payload.get("account_id"))
                if candidate:
                    identifiers.append(candidate)

        for identifier in identifiers:
            _register_account_identifier(mapping, identifier, resolved_entry)

    return mapping


def _resolve_account_dir_map(
    sid: str, runs_root: Path, accounts_dir: Path
) -> dict[str, Path]:
    try:
        marker = accounts_dir.stat().st_mtime
    except OSError:
        marker = -1.0

    key = (runs_root.resolve().as_posix(), sid)
    with _ACCOUNT_MAP_LOCK:
        cached = _ACCOUNT_MAP_CACHE.get(key)
        if cached and cached[0] == marker:
            return cached[1]
        mapping = _build_account_directory_map(accounts_dir)
        _ACCOUNT_MAP_CACHE[key] = (marker, mapping)
        return mapping


_ACCOUNT_ID_DIGITS = re.compile(r"(\d+)")


def _fallback_account_dir(account_id: str, accounts_dir: Path) -> Path | None:
    match = _ACCOUNT_ID_DIGITS.search(account_id)
    if match:
        idx = match.group(1).lstrip("0") or "0"
        candidate = accounts_dir / idx
        if candidate.is_dir():
            return candidate.resolve()

    candidate = accounts_dir / account_id
    if candidate.is_dir():
        return candidate.resolve()

    return None


def _resolve_account_dir(
    account_id: str,
    accounts_dir: Path,
    accounts_map: Mapping[str, Path],
) -> Path | None:
    normalized = normalize_note_style_account_id(account_id)
    for key in (account_id, normalized):
        if not key:
            continue
        candidate = accounts_map.get(key)
        if candidate is not None:
            return candidate
    return _fallback_account_dir(account_id, accounts_dir)


def _load_json_value(path: Path) -> Any | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("NOTE_STYLE_JSON_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("NOTE_STYLE_JSON_PARSE_FAILED path=%s", path, exc_info=True)
        return None


def _load_account_artifacts(
    account_dir: Path | None,
) -> tuple[Mapping[str, Any], dict[str, Mapping[str, Any]], list[Mapping[str, Any]]]:
    if account_dir is None:
        return {}, {}, []

    meta_payload = _load_json_mapping(account_dir / "meta.json")
    if not isinstance(meta_payload, Mapping):
        meta_payload = {}

    raw_bureaus = _load_json_mapping(account_dir / "bureaus.json")
    bureaus: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_bureaus, Mapping):
        for bureau, payload in raw_bureaus.items():
            if isinstance(payload, Mapping):
                bureaus[str(bureau)] = dict(payload)

    raw_tags = _load_json_value(account_dir / "tags.json")
    tags: list[Mapping[str, Any]] = []
    if isinstance(raw_tags, Sequence):
        tags = [entry for entry in raw_tags if isinstance(entry, Mapping)]

    return meta_payload, bureaus, tags


def _coerce_artifact_key(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _register_note_style_case_artifacts(
    *,
    sid: str,
    runs_root: Path,
    account_id: str,
    account_dir: Path | None,
    meta_payload: Mapping[str, Any],
    response_path: Path,
) -> None:
    keys: set[str] = set()
    for candidate in (
        account_id,
        account_dir.name if isinstance(account_dir, Path) else None,
        meta_payload.get("account_id") if isinstance(meta_payload, Mapping) else None,
        meta_payload.get("account_index") if isinstance(meta_payload, Mapping) else None,
    ):
        key = _coerce_artifact_key(candidate)
        if key:
            keys.add(key)

    if not keys:
        return

    artifacts: dict[str, Path] = {
        "note_style_response": response_path.resolve(),
    }
    if isinstance(account_dir, Path):
        resolved_dir = account_dir.resolve()
        artifacts.update(
            {
                "bureaus": resolved_dir / "bureaus.json",
                "meta": resolved_dir / "meta.json",
                "tags": resolved_dir / "tags.json",
            }
        )

    manifest_path = runs_root / sid / "manifest.json"
    try:
        manifest = RunManifest.load_or_create(manifest_path, sid)
    except Exception:  # pragma: no cover - defensive logging
        log.debug(
            "NOTE_STYLE_MANIFEST_ARTIFACT_LOAD_FAILED sid=%s path=%s",
            sid,
            manifest_path,
            exc_info=True,
        )
        return

    for key in sorted(keys):
        group = f"cases.accounts.{key}"
        for name, path in artifacts.items():
            try:
                manifest = manifest.set_artifact(group, name, path)
            except Exception:  # pragma: no cover - defensive logging
                log.debug(
                    "NOTE_STYLE_MANIFEST_ARTIFACT_SET_FAILED sid=%s group=%s name=%s path=%s",
                    sid,
                    group,
                    name,
                    path,
                    exc_info=True,
                )


def _clean_value(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"--", "-", "n/a", "na", "none", "null", "unknown"}:
        return ""
    return text


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return slug


def _collect_issue_tags(tags: Sequence[Mapping[str, Any]]) -> list[str]:
    issues = [
        _clean_value(tag.get("type"))
        for tag in tags
        if isinstance(tag, Mapping)
        and _normalize_text(tag.get("kind")).lower() == "issue"
    ]
    return _unique(issue for issue in issues if issue)


def _collect_other_tags(tags: Sequence[Mapping[str, Any]]) -> list[str]:
    entries: list[str] = []
    for tag in tags:
        if not isinstance(tag, Mapping):
            continue
        kind = _normalize_text(tag.get("kind"))
        if not kind or kind.lower() == "issue":
            continue
        tag_type = _clean_value(tag.get("type"))
        if tag_type:
            entries.append(f"{kind}:{tag_type}")
        else:
            entries.append(kind)
    return _unique(entry for entry in entries if entry)


def _collect_bureau_values(
    bureaus: Mapping[str, Mapping[str, Any]], field: str
) -> list[str]:
    values: list[str] = []
    for _, payload in sorted(bureaus.items(), key=lambda item: item[0]):
        if not isinstance(payload, Mapping):
            continue
        raw_value = payload.get(field)
        if isinstance(raw_value, Mapping):
            for entry in raw_value.values():
                cleaned = _clean_value(entry)
                if cleaned:
                    values.append(cleaned)
        else:
            cleaned = _clean_value(raw_value)
            if cleaned:
                values.append(cleaned)
    return _unique(values)


_BUREAU_FIELD_ORDER: tuple[str, ...] = (
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

_BUREAU_AMOUNT_FIELDS = {
    "balance_owed",
    "high_balance",
    "past_due_amount",
}

_BUREAU_DATE_FIELDS = {
    "date_opened",
    "date_reported",
    "date_of_last_activity",
    "closed_date",
    "last_verified",
}


def _normalize_amount_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        decimal_value = Decimal(str(value))
    elif isinstance(value, Decimal):
        decimal_value = value
    else:
        text = _normalize_text(value)
        if not text:
            return ""
        stripped = text.strip()
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
        stripped = stripped.replace(",", "").replace("$", "").strip()
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


def _normalize_date_value(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
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


def _normalize_bureau_field(field: str, value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("value", "raw", "display", "text"):
            if key in value:
                candidate = _normalize_bureau_field(field, value[key])
                if candidate:
                    return candidate
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for entry in value:
            candidate = _normalize_bureau_field(field, entry)
            if candidate:
                return candidate
        return ""
    if field in _BUREAU_AMOUNT_FIELDS:
        return _normalize_amount_value(value)
    if field in _BUREAU_DATE_FIELDS:
        return _normalize_date_value(value)
    return _clean_value(value)


def _summarize_bureaus(
    bureaus: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(bureaus, Mapping):
        return {}

    per_bureau: dict[str, dict[str, str]] = {}
    field_counts: dict[str, dict[str, int]] = {}
    field_bureau_values: dict[str, dict[str, str]] = {}

    for bureau_name, payload in sorted(bureaus.items(), key=lambda item: item[0]):
        if not isinstance(payload, Mapping):
            continue
        normalized_fields: dict[str, str] = {}
        for field in _BUREAU_FIELD_ORDER:
            normalized = _normalize_bureau_field(field, payload.get(field))
            if not normalized:
                continue
            normalized_fields[field] = normalized
            counts = field_counts.setdefault(field, {})
            counts[normalized] = counts.get(normalized, 0) + 1
            bureau_map = field_bureau_values.setdefault(field, {})
            bureau_map[bureau_name] = normalized
        if normalized_fields:
            per_bureau[bureau_name] = dict(sorted(normalized_fields.items(), key=lambda item: _BUREAU_FIELD_ORDER.index(item[0])))

    if not per_bureau:
        return {}

    majority_values: dict[str, str] = {}
    for field in _BUREAU_FIELD_ORDER:
        counts = field_counts.get(field)
        if not counts:
            continue
        majority_value = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if majority_value:
            majority_values[field] = majority_value

    disagreements: dict[str, dict[str, str]] = {}
    for field in _BUREAU_FIELD_ORDER:
        bureau_map = field_bureau_values.get(field)
        if not bureau_map:
            continue
        unique_values = {value for value in bureau_map.values() if value}
        if len(unique_values) > 1:
            disagreements[field] = dict(sorted(bureau_map.items(), key=lambda item: item[0]))

    return {
        "per_bureau": per_bureau,
        "majority_values": majority_values,
        "disagreements": disagreements,
    }


def _extract_account_tail(
    meta: Mapping[str, Any], bureaus: Mapping[str, Mapping[str, Any]]
) -> str:
    tail = _clean_value(meta.get("account_number_tail"))
    if tail:
        digits = re.sub(r"\D", "", tail)
        return digits[-4:] if digits else tail

    numbers = _collect_bureau_values(bureaus, "account_number_display")
    for number in numbers:
        digits = re.sub(r"\D", "", number)
        if digits:
            return digits[-4:]
    return ""


def _select_primary_issue(tags: Sequence[Mapping[str, Any]]) -> str | None:
    for tag in tags:
        if not isinstance(tag, Mapping):
            continue
        kind = _normalize_text(tag.get("kind"))
        if not kind or kind.lower() != "issue":
            continue
        issue_value = _clean_value(tag.get("type"))
        if issue_value:
            return issue_value
    return None


def _build_account_identity(
    meta: Mapping[str, Any], tags: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    account_id = _clean_value(meta.get("account_id")) or None
    reported_creditor = _clean_value(meta.get("heading_guess")) or None

    identity: dict[str, Any] = {"primary_issue": _select_primary_issue(tags)}
    if account_id is not None:
        identity["account_id"] = account_id
    if reported_creditor is not None:
        identity["reported_creditor"] = reported_creditor

    if len(identity) > 1:
        return identity
    return None


def _build_account_context(
    meta: Mapping[str, Any],
    bureaus: Mapping[str, Mapping[str, Any]],
    tags: Sequence[Mapping[str, Any]],
    bureaus_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {}

    identity = _build_account_identity(meta, tags)
    if identity is not None:
        context["identity"] = identity

    meta_context: dict[str, Any] = {}
    for key in ("heading_guess", "issuer_canonical", "issuer_variant", "issuer_slug"):
        value = _clean_value(meta.get(key))
        if value:
            meta_context[key] = value

    tail = _extract_account_tail(meta, bureaus)
    if tail:
        meta_context["account_number_tail"] = tail

    presence = meta.get("bureau_presence")
    if isinstance(presence, Mapping):
        meta_context["bureau_presence"] = {
            str(bureau): bool(present) for bureau, present in presence.items()
        }

    if meta_context:
        context["meta"] = meta_context

    tags_context: dict[str, Any] = {}
    issues = _collect_issue_tags(tags)
    if issues:
        tags_context["issues"] = issues
    others = _collect_other_tags(tags)
    if others:
        tags_context["other"] = others
    if tags_context:
        context["tags"] = tags_context

    if bureaus_summary is None:
        bureaus_summary = _summarize_bureaus(bureaus)
    if isinstance(bureaus_summary, Mapping) and bureaus_summary.get("per_bureau"):
        context["bureaus"] = bureaus_summary

    return context


def _first_bureau_value(
    bureaus: Mapping[str, Mapping[str, Any]], field: str
) -> str:
    for _, payload in sorted(bureaus.items(), key=lambda item: item[0]):
        if not isinstance(payload, Mapping):
            continue
        candidate = _normalize_bureau_field(field, payload.get(field))
        if candidate:
            return candidate
    return ""


def _has_bureau_conflicts(
    bureaus: Mapping[str, Mapping[str, Any]], fields: Sequence[str]
) -> bool:
    for field in fields:
        values: set[str] = set()
        for payload in bureaus.values():
            if not isinstance(payload, Mapping):
                continue
            candidate = _normalize_bureau_field(field, payload.get(field))
            if candidate:
                values.add(candidate)
        if len(values) > 1:
            return True
    return False


def _build_account_fingerprint(
    account_id: str,
    meta: Mapping[str, Any],
    bureaus: Mapping[str, Mapping[str, Any]],
    tags: Sequence[Mapping[str, Any]],
    bureaus_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    majority_values: Mapping[str, Any] = {}
    if bureaus_summary is None:
        bureaus_summary = _summarize_bureaus(bureaus)
    if isinstance(bureaus_summary, Mapping):
        majority_payload = bureaus_summary.get("majority_values")
        if isinstance(majority_payload, Mapping):
            majority_values = majority_payload
    fingerprint: dict[str, Any] = {}

    normalized_account_id = _clean_value(meta.get("account_id")) or account_id
    account_id_slug = _slugify(normalized_account_id) or "unknown"
    fingerprint["account_id"] = account_id_slug

    identity_section: dict[str, Any] = {}
    reported_creditor = _clean_value(meta.get("heading_guess"))
    if not reported_creditor:
        reported_creditor = _clean_value(majority_values.get("reported_creditor"))
    if not reported_creditor:
        reported_creditor = _first_bureau_value(bureaus, "reported_creditor")
    if reported_creditor:
        identity_section["reported_creditor"] = _slugify(reported_creditor)

    tail = _extract_account_tail(meta, bureaus)
    if tail:
        identity_section["account_tail"] = tail

    if identity_section:
        fingerprint["identity"] = identity_section

    primary_issue = _select_primary_issue(tags)
    if primary_issue:
        fingerprint["core_issue"] = _slugify(primary_issue)

    financial_fields = ("account_type", "account_status", "payment_status")
    financial_section: dict[str, Any] = {}
    for field in financial_fields:
        value = _clean_value(majority_values.get(field))
        if not value:
            value = _first_bureau_value(bureaus, field)
        if value:
            financial_section[field] = _slugify(value)
    if financial_section:
        fingerprint["financial"] = financial_section

    date_section: dict[str, Any] = {}
    opened_value = _clean_value(majority_values.get("date_opened"))
    if not opened_value:
        opened_value = _normalize_date_value(
            _first_bureau_value(bureaus, "date_opened")
        )
    if opened_value:
        date_section["opened"] = opened_value
    last_activity_value = _clean_value(majority_values.get("date_of_last_activity"))
    if not last_activity_value:
        last_activity_value = _normalize_date_value(
            _first_bureau_value(bureaus, "date_of_last_activity")
        )
    if last_activity_value:
        date_section["last_activity"] = last_activity_value
    if date_section:
        fingerprint["dates"] = date_section

    fields_to_compare = (
        "reported_creditor",
        "account_type",
        "account_status",
        "payment_status",
        "date_opened",
        "date_of_last_activity",
    )
    fingerprint["disagreements"] = _has_bureau_conflicts(bureaus, fields_to_compare)

    return fingerprint


def _compute_fingerprint_hash(fingerprint: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        fingerprint, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    digest = hashlib.sha256()
    digest.update(serialized.encode("utf-8"))
    return digest.hexdigest()


def discover_note_style_response_accounts(
    sid: str, *, runs_root: Path | str | None = None
) -> list[NoteStyleResponseAccount]:
    """Return discovered frontend response files for ``sid``.

    The discovery process is read-only. Response files are mapped back to
    their canonical account identifiers and the normalized filenames used by
    the note_style stage are precomputed for convenience.
    """

    runs_root_path = _resolve_runs_root(runs_root)
    source_paths = _resolve_source_paths(sid, runs_root_path)
    responses_dir = source_paths.responses_dir
    if not responses_dir.is_dir():
        log.info("NOTE_STYLE_DISCOVERY sid=%s responses=%s usable=%s", sid, 0, 0)
        return []

    suffix = ".result.json"
    discovered: list[NoteStyleResponseAccount] = []
    responses_total = 0
    usable_total = 0
    accounts_map = _resolve_account_dir_map(sid, runs_root_path, source_paths.accounts_dir)
    for candidate in sorted(responses_dir.glob(f"*{suffix}"), key=lambda path: path.name):
        if not candidate.is_file():
            continue

        responses_total += 1

        try:
            if candidate.stat().st_size == 0:
                log.info(
                    "NOTE_STYLE_DISCOVERY_SKIP_EMPTY sid=%s path=%s",
                    sid,
                    candidate,
                )
                continue
        except FileNotFoundError:
            continue
        except OSError:
            log.warning(
                "NOTE_STYLE_DISCOVERY_STAT_FAILED sid=%s path=%s",
                sid,
                candidate,
                exc_info=True,
            )
            continue

        try:
            raw_payload = candidate.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except OSError:
            log.warning(
                "NOTE_STYLE_DISCOVERY_READ_FAILED sid=%s path=%s",
                sid,
                candidate,
                exc_info=True,
            )
            continue

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            log.warning(
                "NOTE_STYLE_DISCOVERY_INVALID_JSON sid=%s path=%s",
                sid,
                candidate,
                exc_info=True,
            )
            continue

        if not isinstance(payload, Mapping):
            continue

        if not _has_response_note_field(payload):
            log.info(
                "NOTE_STYLE_DISCOVERY_SKIP_NO_NOTE sid=%s path=%s",
                sid,
                candidate,
            )
            continue

        account_id = candidate.name[: -len(suffix)]
        normalized = normalize_note_style_account_id(account_id)
        pack_filename = note_style_pack_filename(account_id)
        result_filename = note_style_result_filename(account_id)
        response_reference = _canonical_response_reference(runs_root_path, candidate)
        account_dir = _resolve_account_dir(
            account_id,
            source_paths.accounts_dir,
            accounts_map,
        )
        discovered.append(
            NoteStyleResponseAccount(
                account_id=account_id,
                normalized_account_id=normalized,
                response_path=candidate.resolve(),
                response_relative=response_reference,
                pack_filename=pack_filename,
                result_filename=result_filename,
                account_dir=account_dir.resolve() if isinstance(account_dir, Path) else None,
            )
        )
        usable_total += 1

    discovered.sort(key=lambda entry: entry.account_id)
    log.info(
        "NOTE_STYLE_DISCOVERY sid=%s responses=%s usable=%s",
        sid,
        responses_total,
        usable_total,
    )
    return discovered


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("NOTE_STYLE_INDEX_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("NOTE_STYLE_INDEX_PARSE_FAILED path=%s", path, exc_info=True)
        return None

    if isinstance(payload, Mapping):
        return payload

    return None


def _load_result_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("NOTE_STYLE_RESULT_READ_FAILED path=%s", path, exc_info=True)
        return None

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None

    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError:
        log.warning("NOTE_STYLE_RESULT_INVALID_JSON path=%s", path, exc_info=True)
        return None

    if isinstance(payload, Mapping):
        return payload

    return None


def _write_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    serialized = json.dumps(row, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    _fsync_directory(path.parent)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(serialized)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    _fsync_directory(path.parent)


def _note_style_log_path(paths: NoteStylePaths) -> Path:
    return getattr(paths, "log_file", paths.base / "logs.txt")


def _validate_account_artifacts(
    *,
    sid: str,
    account_id: str,
    paths: NoteStylePaths,
    account_paths: NoteStyleAccountPaths,
    response: PurePosixPath,
) -> None:
    missing: list[str] = []
    if not account_paths.pack_file.is_file():
        missing.append("pack")
    if not account_paths.result_file.is_file():
        missing.append("result_path")

    if not missing:
        return

    issues = ",".join(sorted(missing))
    log.warning(
        "NOTE_STYLE_ARTIFACT_VALIDATION_FAILED sid=%s account_id=%s response=%s missing=%s",
        sid,
        account_id,
        response,
        issues,
    )
    append_note_style_warning(
        _note_style_log_path(paths),
        f"sid={sid} account_id={account_id} response={response} missing_artifacts={issues}",
    )


def _relativize(path: Path, base: Path) -> str:
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError:
        relative = Path(os.path.relpath(resolved_path, resolved_base))
    return str(PurePosixPath(relative))


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        key = value.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _source_hash(text: str) -> str:
    normalized = " ".join(text.split()).strip().lower()
    digest = hashlib.sha256()
    digest.update(normalized.encode("utf-8"))
    return digest.hexdigest()


def _timeframe_bucket(timeframe: Mapping[str, Any] | None) -> str:
    if not isinstance(timeframe, Mapping):
        return "none"

    relative = _normalize_text(timeframe.get("relative")).lower()
    if relative:
        return f"relative:{relative}"

    month = _normalize_text(timeframe.get("month"))
    if month:
        return f"month:{month}"

    return "none"


def _amount_band(entities: Mapping[str, Any] | None) -> str:
    amount: Any | None = None
    if isinstance(entities, Mapping):
        amount = entities.get("amount")

    try:
        value = float(amount)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "none"

    if not math.isfinite(value):
        return "none"

    absolute = abs(value)
    if absolute == 0:
        return "zero"
    if absolute < 100:
        return "lt_100"
    if absolute < 500:
        return "100_499"
    if absolute < 1000:
        return "500_999"
    if absolute < 5000:
        return "1000_4999"
    if absolute < 10000:
        return "5000_9999"
    return "gte_10000"


def _sorted_emphasis(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []

    normalized: set[str] = set()
    for entry in values:
        text = _normalize_text(entry)
        if text:
            normalized.add(text)

    return sorted(normalized)


def _analysis_summary(
    payload: Mapping[str, Any] | None,
) -> tuple[str, str, list[str], float | None, list[str]]:
    tone = ""
    topic = ""
    emphasis: list[str] = []
    confidence_value: float | None = None
    risk_flags: list[str] = []

    if isinstance(payload, Mapping):
        tone = _normalize_text(payload.get("tone"))

        context = payload.get("context_hints")
        if isinstance(context, Mapping):
            topic = _normalize_text(context.get("topic"))

        emphasis_values = payload.get("emphasis")
        if isinstance(emphasis_values, Sequence) and not isinstance(
            emphasis_values, (str, bytes, bytearray)
        ):
            emphasis = _unique(_normalize_text(value) for value in emphasis_values)

        confidence_raw = payload.get("confidence")
        try:
            confidence_candidate = float(confidence_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            confidence_candidate = None
        else:
            if math.isfinite(confidence_candidate):
                confidence_value = confidence_candidate

        risk_values = payload.get("risk_flags")
        if isinstance(risk_values, Sequence) and not isinstance(
            risk_values, (str, bytes, bytearray)
        ):
            risk_flags = _unique(_normalize_text(value) for value in risk_values)

    return tone, topic, emphasis, confidence_value, risk_flags


def _log_style_discovery(
    *,
    sid: str,
    account_id: str,
    response: PurePosixPath,
    status: str,
    note_hash: str | None = None,
    source_hash: str | None = None,
    char_len: int | None = None,
    word_len: int | None = None,
    truncated: bool | None = None,
    analysis: Mapping[str, Any] | None = None,
    prompt_salt: str | None = None,
    reason: str | None = None,
) -> None:
    tone, topic, emphasis_values, confidence_value, risk_flags = _analysis_summary(analysis)
    confidence_text = ""
    if confidence_value is not None:
        confidence_text = f"{confidence_value:.2f}"

    log.info(
        "NOTE_STYLE_DISCOVERY_DETAIL sid=%s account_id=%s response=%s status=%s note_hash=%s source_hash=%s chars=%s words=%s truncated=%s tone=%s topic=%s emphasis=%s confidence=%s risk_flags=%s prompt_salt=%s reason=%s",
        sid,
        account_id,
        str(response),
        status,
        note_hash or "",
        source_hash or "",
        "" if char_len is None else char_len,
        "" if word_len is None else word_len,
        "" if truncated is None else truncated,
        tone,
        topic,
        "|".join(emphasis_values),
        confidence_text,
        "|".join(risk_flags),
        prompt_salt or "",
        reason or "",
    )


def _note_hash(note_text: str) -> str:
    normalized = _collapse_whitespace(unicodedata.normalize("NFKC", note_text))
    digest = hashlib.sha256()
    digest.update(normalized.encode("utf-8"))
    return digest.hexdigest()


def _prepare_note_text_for_model(note_text: str) -> tuple[str, bool]:
    normalized = unicodedata.normalize("NFKC", note_text)
    masked = _mask_contact_info(normalized)
    sanitized = _collapse_whitespace(masked)
    truncated = False
    if len(sanitized) > _NOTE_TEXT_MAX_CHARS:
        sanitized = sanitized[:_NOTE_TEXT_MAX_CHARS]
        truncated = True
    return sanitized, truncated


def _random_prompt_salt() -> str:
    if config.NOTE_STYLE_DISABLE_SALT:
        return ""

    length = secrets.choice(range(8, 13))
    bytes_needed = math.ceil(length / 2)
    value = secrets.token_hex(bytes_needed)
    return value[:length]


def _pack_messages(
    *,
    sid: str,
    account_id: str,
    note_text: str,
    note_truncated: bool,
    prompt_salt: str | None,
    fingerprint_hash: str,
    account_context: Mapping[str, Any] | None,
    bureaus_summary: Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    system_message = _NOTE_STYLE_SYSTEM_PROMPT.format(prompt_salt=prompt_salt or "")
    metadata: dict[str, Any] = {
        "sid": sid,
        "account_id": account_id,
        "fingerprint_hash": fingerprint_hash,
        "channel": "frontend_review",
        "lang": "auto",
    }

    user_content: dict[str, Any] = {
        "note_text": note_text,
        "note_truncated": note_truncated,
        "metadata": metadata,
    }

    if account_context:
        user_content["account_context"] = account_context
    if bureaus_summary:
        user_content["bureaus_summary"] = bureaus_summary

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _collapse_whitespace(value: str) -> str:
    translated = value.translate(_ZERO_WIDTH_WHITESPACE)
    return " ".join(translated.split()).strip()


def _mask_contact_info(value: str) -> str:
    if not value:
        return value

    masked = _EMAIL_PATTERN.sub("[redacted_email]", value)
    for pattern in _PHONE_PATTERNS:
        masked = pattern.sub("[redacted_phone]", masked)
    return masked


def _normalize_low_signal_candidate(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    return " ".join(normalized.split())


def _is_low_signal_note(note: str) -> bool:
    normalized = _normalize_low_signal_candidate(note)
    if not normalized:
        return True
    if len(normalized) <= 3:
        return True
    return normalized in _LOW_SIGNAL_PHRASES


def _sanitize_note_value(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    normalized = unicodedata.normalize("NFKC", value)
    masked = _mask_contact_info(normalized)
    return _collapse_whitespace(masked)


def _iter_note_value_candidates(root: Any, parts: Sequence[str]) -> Iterator[Any]:
    if root is None:
        return
    if not parts:
        yield root
        return

    head, *tail = parts
    if isinstance(root, Mapping):
        value = root.get(head)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for entry in value:
                yield from _iter_note_value_candidates(entry, tail)
        else:
            yield from _iter_note_value_candidates(value, tail)
        return

    if isinstance(root, Sequence) and not isinstance(root, (str, bytes, bytearray)):
        for entry in root:
            yield from _iter_note_value_candidates(entry, parts)


def _flatten_note_candidate(value: Any) -> Iterator[tuple[str, str]]:
    if value is None:
        return
    if isinstance(value, str):
        sanitized = _sanitize_note_value(value)
        yield value, sanitized
        return
    if isinstance(value, Mapping):
        for entry in value.values():
            yield from _flatten_note_candidate(entry)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for entry in value:
            yield from _flatten_note_candidate(entry)
        return

    sanitized = _sanitize_note_value(value)
    raw_text = value if isinstance(value, str) else str(value)
    yield raw_text, sanitized


def _has_response_note_field(payload: Mapping[str, Any]) -> bool:
    for path in _NOTE_VALUE_PATHS:
        for candidate in _iter_note_value_candidates(payload, path):
            for _raw_value, _sanitized in _flatten_note_candidate(candidate):
                return True
    return False


def _extract_note_text(payload: Mapping[str, Any]) -> tuple[str, str]:
    fallback_raw: str | None = None
    for path in _NOTE_VALUE_PATHS:
        for candidate in _iter_note_value_candidates(payload, path):
            for raw_value, sanitized in _flatten_note_candidate(candidate):
                if len(sanitized) > 0:
                    return raw_value, sanitized
                if fallback_raw is None and isinstance(raw_value, str):
                    fallback_raw = raw_value

    if fallback_raw is not None:
        return fallback_raw, ""

    return "", ""


def _extract_ui_allegations_selected(payload: Mapping[str, Any]) -> tuple[str, ...]:
    results: list[str] = []
    seen: set[str] = set()

    def _register(value: Any) -> None:
        if isinstance(value, str):
            candidate = _normalize_text(value)
            if not candidate:
                return
            normalized = candidate.lower()
            if normalized in seen:
                return
            seen.add(normalized)
            results.append(candidate)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for entry in value:
                _register(entry)
            return
        if isinstance(value, Mapping):
            for entry in value.values():
                _register(entry)

    def _walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if _normalize_text(key).lower() == "ui_allegations_selected":
                    _register(value)
                else:
                    _walk(value)
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for entry in node:
                _walk(entry)

    _walk(payload)
    return tuple(results)


def _tokens(note: str) -> set[str]:
    cleaned = note.replace("/", " ").replace("-", " ")
    return {token for token in cleaned.lower().split() if token}


def _contains_phrase(text: str, *phrases: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _normalize_month_value(name: str | None, year_text: str | None) -> str | None:
    if not name:
        return None
    key = name.lower().strip(".")
    month = _MONTH_NAME_MAP.get(key)
    if not month:
        return None
    if not year_text:
        return month
    try:
        year = int(year_text)
        if year < 100:
            year += 2000 if year < 50 else 1900
        return f"{month}-{year}"
    except ValueError:
        return month


def _extract_timeframe(note: str) -> dict[str, str | None]:
    month: str | None = None
    relative: str | None = None

    match = _MONTH_PATTERN.search(note)
    if match:
        month = _normalize_month_value(match.group(1), match.group(2))

    lowered = note.lower()
    for key, patterns in _RELATIVE_TIMEFRAME_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                relative = key
                break
        if relative:
            break

    return {"month": month, "relative": relative}


def _extract_entities(note: str) -> tuple[dict[str, Any], list[str]]:
    entities: dict[str, Any] = {"creditor": None, "amount": None}
    lowered = note.lower()
    risk_flags: list[str] = []

    for key, display in _KNOWN_CREDITORS.items():
        if key in lowered:
            entities["creditor"] = display
            break

    if entities["creditor"] is None:
        match = _CREDITOR_PATTERN.search(note)
        if match:
            candidate = match.group(1).strip()
            words = [word for word in candidate.split() if len(word) > 1]
            if words:
                entities["creditor"] = " ".join(word.strip().title() for word in words)

    amount_match = _AMOUNT_PATTERN.search(note.replace(",", ""))
    if amount_match:
        amount_text = amount_match.group(0).replace("$", "")
        try:
            amount_value = float(amount_text)
        except ValueError:
            amount_value = None
        if amount_value is not None:
            entities["amount"] = round(amount_value, 2)
            if amount_value >= 10000:
                risk_flags.append("large_amount")

    return entities, risk_flags


def _detect_personal_data(note: str) -> bool:
    lowered = note.lower()
    if "social security" in lowered or "ssn" in lowered:
        return True
    if "date of birth" in lowered or "dob" in lowered:
        return True
    for pattern in _PERSONAL_DATA_PATTERNS:
        if pattern.search(note):
            return True
    return False


def _tone_from_note(note: str, tokens: set[str]) -> tuple[str, float, list[str]]:
    text = note.lower()
    exclamations = note.count("!")
    risk_flags: list[str] = []

    if any(word in tokens for word in {"urgent", "immediately", "asap", "now"}) or exclamations >= 2:
        tone = "assertive"
        confidence = 0.82
    elif _contains_phrase(text, "please help", "please assist", "thank you") or {
        "please",
        "help",
    }.issubset(tokens):
        tone = "empathetic"
        confidence = 0.74
    elif _contains_phrase(text, "i dispute", "this is incorrect", "not accurate") or "dispute" in tokens:
        tone = "confident"
        confidence = 0.72
    elif _contains_phrase(text, "i am requesting", "i am writing") or "sincerely" in tokens:
        tone = "formal"
        confidence = 0.7
    elif len(re.findall(r"\d", note)) >= 6 and exclamations == 0:
        tone = "factual"
        confidence = 0.68
    elif _contains_phrase(text, "just wanted") or "hey" in tokens:
        tone = "conversational"
        confidence = 0.65
    elif any(word in tokens for word in {"calm", "appreciate", "understand"}) and exclamations == 0:
        tone = "calm"
        confidence = 0.64
    else:
        tone = "neutral"
        confidence = 0.45

    if any(word in tokens for word in {"lawsuit", "court", "legal", "attorney"}):
        risk_flags.append("legal_threat")
    if exclamations >= 3:
        risk_flags.append("escalation_risk")

    if tone not in _ALLOWED_TONES:
        tone = "neutral"

    return tone, confidence, _unique(risk_flags)


def _topic_from_note(note: str, tokens: set[str]) -> tuple[str, float]:
    text = note.lower()
    if _contains_phrase(text, "identity theft", "identity fraud"):
        return "identity_theft", 0.85
    if _contains_phrase(text, "not mine", "never opened", "unauthorized"):
        return "not_mine", 0.82
    if _contains_phrase(text, "billing error", "charged the wrong", "wrong amount") or (
        "billing" in tokens and "error" in tokens
    ):
        return "billing_error", 0.8
    if _contains_phrase(text, "late fee", "late fees"):
        return "late_fee", 0.75
    if any(
        word in tokens
        for word in {"paid", "payment", "balance", "settled", "already", "dispute", "disputed", "disputing"}
    ):
        return "payment_dispute", 0.78
    return "other", 0.45


def _emphasis_from_note(note: str, tokens: set[str]) -> tuple[list[str], float, list[str]]:
    text = note.lower()
    emphasis: list[str] = []
    risk_flags: list[str] = []

    if "paid" in tokens and ("already" in tokens or _contains_phrase(text, "already paid", "paid in full")):
        emphasis.append("paid_already")
    if any(word in tokens for word in {"incorrect", "inaccurate", "error", "wrong", "mistake"}):
        emphasis.append("inaccurate_reporting")
    if _contains_phrase(text, "identity theft", "identity fraud") or "fraud" in tokens:
        emphasis.append("identity_concerns")
        risk_flags.append("identity_theft_claim")
    if _contains_phrase(text, "please help", "need assistance", "need help"):
        emphasis.append("support_request")
    if ("late" in tokens and "fee" in tokens) or _contains_phrase(text, "late fee"):
        emphasis.append("fee_waiver")
    if _contains_phrase(text, "not mine", "never opened", "unauthorized"):
        emphasis.append("ownership_dispute")
    if any(word in tokens for word in {"update", "correct", "fix", "remove", "delete"}):
        emphasis.append("update_requested")
    if any(word in tokens for word in {"attached", "documents", "proof", "evidence"}):
        emphasis.append("evidence_provided")

    filtered = [value for value in _unique(emphasis) if value in _ALLOWED_EMPHASIS]
    confidence = 0.4 if not filtered else min(0.85, 0.55 + 0.07 * len(filtered))
    return filtered, confidence, _unique(risk_flags)


def _build_extractor(note: str) -> dict[str, Any]:
    tokens = _tokens(note)
    tone, tone_confidence, tone_risks = _tone_from_note(note, tokens)
    topic, topic_confidence = _topic_from_note(note, tokens)
    emphasis_values, emphasis_confidence, emphasis_risks = _emphasis_from_note(note, tokens)
    timeframe = _extract_timeframe(note)
    entities, entity_risks = _extract_entities(note)

    risk_flags = set(tone_risks) | set(emphasis_risks) | set(entity_risks)
    if _detect_personal_data(note):
        risk_flags.add("personal_data")

    confidence = 0.3
    if tone != "neutral":
        confidence += min(0.25, tone_confidence * 0.3)
    if topic != "other":
        confidence += min(0.2, topic_confidence * 0.2)
    if emphasis_values:
        confidence += min(0.25, emphasis_confidence * 0.2 + 0.05 * len(emphasis_values))
    if timeframe.get("month") or timeframe.get("relative"):
        confidence += 0.05
    if entities.get("amount") is not None or entities.get("creditor"):
        confidence += 0.05

    confidence = round(min(confidence, 0.95), 2)

    if confidence < 0.5:
        tone = "neutral"
        emphasis_values = []
        topic = "other"

    if tone not in _ALLOWED_TONES:
        tone = "neutral"
    if topic not in _ALLOWED_TOPICS:
        topic = "other"

    emphasis_values = [value for value in emphasis_values if value in _ALLOWED_EMPHASIS]

    if "personal_data" in risk_flags:
        entities = {"creditor": None, "amount": None}

    extractor = {
        "tone": tone,
        "context_hints": {
            "timeframe": {
                "month": timeframe.get("month"),
                "relative": timeframe.get("relative"),
            },
            "topic": topic,
            "entities": {
                "creditor": entities.get("creditor"),
                "amount": entities.get("amount"),
            },
        },
        "emphasis": emphasis_values,
        "confidence": confidence,
        "risk_flags": sorted(risk_flags),
    }

    return extractor


def _load_response_note(account_id: str, response_path: Path) -> _LoadedResponseNote:
    try:
        stats = response_path.stat()
    except FileNotFoundError:
        raise NoteStyleSkip("missing_response") from None
    except OSError:
        log.warning(
            "NOTE_STYLE_RESPONSE_STAT_FAILED path=%s", response_path, exc_info=True
        )
        raise NoteStyleSkip("response_read_failed") from None

    if stats.st_size == 0:
        empty = ""
        return _LoadedResponseNote(
            account_id=str(account_id),
            note_raw=empty,
            note_sanitized="",
            source_path=response_path,
            source_hash=_source_hash(empty),
            ui_allegations_selected=(),
        )

    try:
        raw = response_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise NoteStyleSkip("missing_response") from None
    except OSError:
        log.warning("NOTE_STYLE_RESPONSE_READ_FAILED path=%s", response_path, exc_info=True)
        raise NoteStyleSkip("response_read_failed") from None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("NOTE_STYLE_RESPONSE_INVALID_JSON path=%s", response_path, exc_info=True)
        raise NoteStyleSkip("invalid_response") from None

    if not isinstance(payload, Mapping):
        raise NoteStyleSkip("invalid_response")

    note_raw, note_sanitized = _extract_note_text(payload)
    ui_allegations_selected = _extract_ui_allegations_selected(payload)
    source_hash = _source_hash(note_sanitized)
    return _LoadedResponseNote(
        account_id=str(account_id),
        note_raw=note_raw,
        note_sanitized=note_sanitized,
        source_path=response_path,
        source_hash=source_hash,
        ui_allegations_selected=ui_allegations_selected,
    )


def _index_items(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    packs = payload.get("packs")
    if isinstance(packs, Sequence):
        return [dict(entry) for entry in packs if isinstance(entry, Mapping)]
    items = payload.get("items")
    if isinstance(items, Sequence):
        return [dict(entry) for entry in items if isinstance(entry, Mapping)]
    return []


def _serialize_entry(
    *,
    sid: str,
    account_id: str,
    paths: NoteStylePaths,
    account_paths: NoteStyleAccountPaths,
    note_hash: str,
    status: str,
    timestamp: str,
    result_path: Path | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "account_id": account_id,
        "pack": _relativize(account_paths.pack_file, paths.base),
        "lines": 1,
        "built_at": timestamp,
        "status": status,
        "note_hash": note_hash,
    }
    if result_path is not None:
        entry["result_path"] = _relativize(result_path, paths.base)
    else:
        entry["result_path"] = ""
    return entry


def _serialize_skip_entry(
    *, account_id: str, note_hash: str, timestamp: str, status: str = "skipped_low_signal"
) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "pack": "",
        "lines": 0,
        "built_at": timestamp,
        "status": status,
        "note_hash": note_hash,
        "result_path": "",
    }


def _compute_totals(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total = 0
    completed = 0
    failed = 0
    for entry in items:
        status = _normalize_text(entry.get("status")).lower()
        if status in {"", "skipped", "skipped_low_signal"}:
            continue
        total += 1
        if status in {"completed", "success"}:
            completed += 1
        elif status in {"failed", "error"}:
            failed += 1
    if total == 0 and completed == 0 and failed == 0:
        return {"total": 0, "completed": 0, "failed": 0}
    return {"total": total, "completed": completed, "failed": failed}


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _fsync_directory(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


@contextmanager
def _index_lock(index_path: Path) -> Iterator[None]:
    """Serialize index writers to avoid concurrent clobbering."""

    index_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = index_path.with_suffix(index_path.suffix + ".lock")

    if fcntl is not None:  # pragma: no branch - preferred path on POSIX
        with lock_path.open("a+") as handle:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    break
                except InterruptedError:
                    continue
            try:
                yield
            finally:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
        return

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                stats = lock_path.stat()
            except FileNotFoundError:
                continue
            if (time.time() - stats.st_mtime) > _INDEX_LOCK_STALE_TIMEOUT:
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    continue
                continue
            time.sleep(_INDEX_LOCK_POLL_INTERVAL)

    try:
        os.close(fd)
    except OSError:
        pass

    try:
        yield
    finally:
        try:
            os.unlink(lock_path)
        except FileNotFoundError:
            pass


def _atomic_write_index(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    try:
        with tmp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    _fsync_directory(path.parent)


def _write_index(
    *,
    sid: str,
    paths: NoteStylePaths,
    items: Sequence[Mapping[str, Any]],
) -> None:
    document = {
        "schema_version": _INDEX_SCHEMA_VERSION,
        "sid": sid,
        "root": ".",
        "packs_dir": _relativize(paths.packs_dir, paths.base),
        "results_dir": _relativize(paths.results_dir, paths.base),
        "packs": list(items),
    }
    _atomic_write_index(paths.index_file, document)


def _remove_account_artifacts(account_paths: NoteStyleAccountPaths) -> None:
    for path in (
        account_paths.pack_file,
        account_paths.result_file,
        account_paths.debug_file,
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            log.warning("NOTE_STYLE_ARTIFACT_REMOVE_FAILED path=%s", path, exc_info=True)


def _update_index_for_account(
    *,
    sid: str,
    paths: NoteStylePaths,
    account_id: str,
    entry: Mapping[str, Any] | None,
) -> dict[str, int]:
    index_path = paths.index_file
    replaced_flag = False
    created_flag = False
    removed_flag = False
    removed_entry: Mapping[str, Any] | None = None
    with _index_lock(index_path):
        existing = _load_json_mapping(index_path)
        items = _index_items(existing)

        normalized_account = str(account_id)
        rewritten: list[dict[str, Any]] = []
        for item in items:
            if str(item.get("account_id")) == normalized_account:
                if entry is not None:
                    rewritten.append(dict(entry))
                    replaced_flag = True
                else:
                    removed_flag = True
                    removed_entry = dict(item)
                continue
            rewritten.append(item)

        if entry is not None and not replaced_flag:
            rewritten.append(dict(entry))
            created_flag = True

        rewritten.sort(key=lambda item: str(item.get("account_id", "")))
        _write_index(sid=sid, paths=paths, items=rewritten)
        totals = _compute_totals(rewritten)

    if entry is None:
        if removed_flag:
            action = "removed"
        else:
            action = "noop"
        status_text = "removed" if removed_flag else ""
        pack_value = str((removed_entry or {}).get("pack") or "")
        result_value = str((removed_entry or {}).get("result_path") or "")
    else:
        action = "updated" if replaced_flag else "created" if created_flag else "noop"
        status_text = _normalize_text(entry.get("status")) if isinstance(entry, Mapping) else ""
        pack_value = str(entry.get("pack") or "")
        result_value = str(entry.get("result_path") or "")

    index_relative = _relativize(paths.index_file, paths.base)
    note_hash_value = ""
    if entry is not None and isinstance(entry, Mapping):
        note_hash_value = str(entry.get("note_hash") or entry.get("source_hash") or "")
    elif removed_entry is not None:
        note_hash_value = str(
            removed_entry.get("note_hash")
            or removed_entry.get("source_hash")
            or ""
        )
    log.info(
        "NOTE_STYLE_INDEX_UPDATED sid=%s account_id=%s action=%s status=%s packs_total=%s packs_completed=%s packs_failed=%s index=%s pack=%s result_path=%s note_hash=%s",
        sid,
        account_id,
        action,
        status_text,
        totals.get("total", 0),
        totals.get("completed", 0),
        totals.get("failed", 0),
        index_relative,
        pack_value,
        result_value,
        note_hash_value,
    )

    return totals


def _note_style_index_progress(index_path: Path) -> tuple[int, int, int, int]:
    document = _load_json_mapping(index_path)
    if not isinstance(document, Mapping):
        return (0, 0, 0, 0)

    entries: Sequence[Mapping[str, Any]] = ()
    packs_payload = document.get("packs")
    if isinstance(packs_payload, Sequence):
        entries = [entry for entry in packs_payload if isinstance(entry, Mapping)]
    else:
        items_payload = document.get("items")
        if isinstance(items_payload, Sequence):
            entries = [entry for entry in items_payload if isinstance(entry, Mapping)]

    total = 0
    completed = 0
    failed = 0
    skipped = 0

    for entry in entries:
        status_text = _normalize_text(entry.get("status")).lower()
        if status_text in {"", "skipped", "skipped_low_signal"}:
            if status_text in {"skipped", "skipped_low_signal"}:
                skipped += 1
            continue
        total += 1
        if status_text == "completed":
            completed += 1
        elif status_text in {"failed", "error"}:
            failed += 1

    if not entries:
        totals_payload = document.get("totals")
        if isinstance(totals_payload, Mapping):
            total = _coerce_int(totals_payload.get("total")) or 0
            completed = _coerce_int(totals_payload.get("completed")) or 0
            failed = _coerce_int(totals_payload.get("failed")) or 0

    return (max(total, 0), max(completed, 0), max(failed, 0), max(skipped, 0))


def _record_stage_progress(
    *, sid: str, runs_root: Path, totals: Mapping[str, int], index_path: Path
) -> None:
    packs_total, packs_completed, packs_failed, packs_skipped = _note_style_index_progress(
        index_path
    )

    if packs_total > 0 and packs_failed > 0:
        status: str = "error"
    elif packs_total == 0:
        status = "success"
    elif packs_completed == packs_total:
        status = "success"
    else:
        status = "built"

    empty_ok = packs_total == 0
    ready = status == "success"

    counts = {"packs_total": packs_total}
    metrics = {"packs_total": packs_total}
    results = {
        "results_total": packs_total,
        "completed": packs_completed,
        "failed": packs_failed,
    }

    manifest_timestamp: str | None = None
    if status in {"built", "success"}:
        manifest_timestamp = _now_iso()
        try:
            register_note_style_build(
                sid,
                runs_root=runs_root,
                timestamp=manifest_timestamp,
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "NOTE_STYLE_MANIFEST_UPDATE_FAILED sid=%s", sid, exc_info=True
            )

    log.info(
        "NOTE_STYLE_REFRESH sid=%s ready=%s total=%s completed=%s failed=%s skipped=%s",
        sid,
        ready,
        packs_total,
        packs_completed,
        packs_failed,
        packs_skipped,
    )
    log_structured_event(
        "NOTE_STYLE_REFRESH",
        logger=log,
        sid=sid,
        ready=ready,
        status=status,
        packs_total=packs_total,
        packs_completed=packs_completed,
        packs_failed=packs_failed,
        packs_skipped=packs_skipped,
        manifest_timestamp=manifest_timestamp,
    )

    record_stage(
        sid,
        "note_style",
        status=status,
        counts=counts,
        empty_ok=empty_ok,
        metrics=metrics,
        results=results,
        runs_root=runs_root,
    )


def build_note_style_pack_for_account(
    sid: str, account_id: str, *, runs_root: Path | str | None = None
) -> Mapping[str, Any]:
    runs_root_path = _resolve_runs_root(runs_root)
    source_paths = _resolve_source_paths(sid, runs_root_path)
    ensure_note_style_section(sid, runs_root=runs_root_path)
    account_id_str = str(account_id)
    response_path = source_paths.responses_dir / f"{account_id}.result.json"
    response_rel = _canonical_response_reference(runs_root_path, response_path)

    paths = ensure_note_style_paths(runs_root_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    accounts_map = _resolve_account_dir_map(
        sid, runs_root_path, source_paths.accounts_dir
    )
    account_dir = _resolve_account_dir(
        account_id_str,
        source_paths.accounts_dir,
        accounts_map,
    )
    meta_payload, bureaus_payload, tags_payload = _load_account_artifacts(account_dir)
    _register_note_style_case_artifacts(
        sid=sid,
        runs_root=runs_root_path,
        account_id=account_id_str,
        account_dir=account_dir,
        meta_payload=meta_payload,
        response_path=response_path,
    )

    try:
        loaded_note = _load_response_note(account_id, response_path)
    except NoteStyleSkip as exc:
        reason = exc.reason or "error"
        totals = _update_index_for_account(sid=sid, paths=paths, account_id=account_id, entry=None)
        _remove_account_artifacts(account_paths)
        _record_stage_progress(
            sid=sid, runs_root=runs_root_path, totals=totals, index_path=paths.index_file
        )
        _log_style_discovery(
            sid=sid,
            account_id=account_id_str,
            response=response_rel,
            status="skipped",
            reason=reason,
        )
        return {
            "status": "skipped",
            "reason": reason,
            "packs_total": totals.get("total", 0),
            "packs_completed": totals.get("completed", 0),
        }

    note_sanitized = loaded_note.note_sanitized
    note_text_for_model, note_truncated = _prepare_note_text_for_model(note_sanitized)
    source_hash = loaded_note.source_hash
    ui_allegations_selected = list(loaded_note.ui_allegations_selected)
    note_hash = _note_hash(note_sanitized)
    char_len = len(note_text_for_model)
    word_len = len(note_text_for_model.split())
    if _is_low_signal_note(note_text_for_model):
        timestamp = _now_iso()
        skip_status = "skipped_low_signal"
        _remove_account_artifacts(account_paths)
        entry = _serialize_skip_entry(
            account_id=account_id_str,
            note_hash=note_hash,
            timestamp=timestamp,
            status=skip_status,
        )
        totals = _update_index_for_account(
            sid=sid, paths=paths, account_id=account_id, entry=entry
        )
        _record_stage_progress(
            sid=sid, runs_root=runs_root_path, totals=totals, index_path=paths.index_file
        )
        log.info(
            "NOTE_STYLE_PACK_SKIPPED_LOW_SIGNAL sid=%s acc=%s note_hash=%s char_len=%s word_len=%s truncated=%s",
            sid,
            account_id_str,
            note_hash,
            char_len,
            word_len,
            note_truncated,
        )
        _log_style_discovery(
            sid=sid,
            account_id=account_id_str,
            response=response_rel,
            status=skip_status,
            note_hash=note_hash,
            source_hash=source_hash,
            char_len=char_len,
            word_len=word_len,
            truncated=note_truncated,
            reason="low_signal",
        )
        return {
            "status": skip_status,
            "reason": "low_signal",
            "note_hash": note_hash,
            "packs_total": totals.get("total", 0),
            "packs_completed": totals.get("completed", 0),
        }
    existing_index = _load_json_mapping(paths.index_file)
    items = _index_items(existing_index)
    existing_entry: Mapping[str, Any] | None = None
    for item in items:
        if str(item.get("account_id")) == str(account_id):
            existing_entry = item
            break

    existing_result = _load_result_payload(account_paths.result_file)
    existing_note_hash = ""
    if isinstance(existing_entry, Mapping):
        existing_note_hash = str(
            existing_entry.get("note_hash")
            or existing_entry.get("source_hash")
            or ""
        )
    result_note_hash = ""
    result_source_hash = ""
    if isinstance(existing_result, Mapping):
        result_note_hash = str(existing_result.get("note_hash") or "")
        result_source_hash = str(existing_result.get("source_hash") or "")
    skip_by_note_hash = (
        config.NOTE_STYLE_IDEMPOTENT_BY_NOTE_HASH
        and existing_note_hash == note_hash
        and isinstance(existing_result, Mapping)
        and (
            result_note_hash == note_hash
            or (result_source_hash and result_source_hash == source_hash)
        )
    )
    skip_by_existing_result = (
        not skip_by_note_hash
        and config.NOTE_STYLE_SKIP_IF_RESULT_EXISTS
        and isinstance(existing_result, Mapping)
    )

    if skip_by_note_hash or skip_by_existing_result:
        totals = _compute_totals(items)
        skip_status = "unchanged" if skip_by_note_hash else "existing_result"
        _record_stage_progress(
            sid=sid,
            runs_root=runs_root_path,
            totals=totals,
            index_path=paths.index_file,
        )
        prompt_salt_existing = (
            str(existing_result.get("prompt_salt")) if isinstance(existing_result, Mapping) else ""
        )
        _log_style_discovery(
            sid=sid,
            account_id=account_id_str,
            response=response_rel,
            status=skip_status,
            note_hash=note_hash,
            source_hash=source_hash,
            char_len=char_len,
            word_len=word_len,
            truncated=note_truncated,
            prompt_salt=prompt_salt_existing,
        )
        return {
            "status": skip_status,
            "packs_total": totals.get("total", 0),
            "packs_completed": totals.get("completed", 0),
        }

    prompt_salt = _random_prompt_salt()
    _log_style_discovery(
        sid=sid,
        account_id=account_id_str,
        response=response_rel,
        status="ready",
        note_hash=note_hash,
        source_hash=source_hash,
        char_len=char_len,
        word_len=word_len,
        truncated=note_truncated,
        prompt_salt=prompt_salt,
    )
    timestamp = _now_iso()

    debug_snapshot = {
        "sid": sid,
        "account_id": account_id_str,
        "collected_at": timestamp,
        "fingerprint": None,
        "fingerprint_hash": "",
        "meta": meta_payload,
        "bureaus": bureaus_payload,
        "bureaus_summary": None,
        "tags": tags_payload,
    }

    if account_dir is None:
        log.info(
            "NOTE_STYLE_ACCOUNT_DIR_MISSING sid=%s acc=%s",
            sid,
            account_id_str,
        )

    bureaus_summary = _summarize_bureaus(bureaus_payload)
    account_context = _build_account_context(
        meta_payload, bureaus_payload, tags_payload, bureaus_summary
    )
    fingerprint = _build_account_fingerprint(
        account_id_str,
        meta_payload,
        bureaus_payload,
        tags_payload,
        bureaus_summary,
    )
    fingerprint_hash = _compute_fingerprint_hash(fingerprint)

    debug_snapshot.update(
        {
            "fingerprint": fingerprint,
            "fingerprint_hash": fingerprint_hash,
            "bureaus_summary": bureaus_summary,
            "account_context": account_context,
        }
    )
    _write_json(account_paths.debug_file, debug_snapshot)

    pack_payload = {
        "sid": sid,
        "account_id": str(account_id),
        "source_response_path": str(response_rel),
        "note_hash": note_hash,
        "model": _NOTE_STYLE_MODEL,
        "prompt_salt": prompt_salt,
        "fingerprint": fingerprint,
        "fingerprint_hash": fingerprint_hash,
        "account_context": account_context,
        "bureaus_summary": bureaus_summary,
        "note_metrics": {"char_len": char_len, "word_len": word_len},
        "messages": _pack_messages(
            sid=sid,
            account_id=str(account_id),
            note_text=note_text_for_model,
            note_truncated=note_truncated,
            prompt_salt=prompt_salt,
            fingerprint_hash=fingerprint_hash,
            account_context=account_context,
            bureaus_summary=bureaus_summary,
        ),
        "built_at": timestamp,
    }
    result_payload = {
        "sid": sid,
        "account_id": str(account_id),
        "prompt_salt": prompt_salt,
        "note_hash": note_hash,
        "note_metrics": {
            "char_len": char_len,
            "word_len": word_len,
            "truncated": note_truncated,
        },
        "evaluated_at": timestamp,
        "fingerprint_hash": fingerprint_hash,
        "account_context": account_context,
        "bureaus_summary": bureaus_summary,
    }

    if ui_allegations_selected:
        pack_payload["ui_allegations_selected"] = ui_allegations_selected
        result_payload["ui_allegations_selected"] = ui_allegations_selected

    _write_jsonl(account_paths.pack_file, pack_payload)
    _write_jsonl(account_paths.result_file, result_payload)

    _validate_account_artifacts(
        sid=sid,
        account_id=account_id_str,
        paths=paths,
        account_paths=account_paths,
        response=response_rel,
    )

    pack_relative = _relativize(account_paths.pack_file, paths.base)
    result_relative = _relativize(account_paths.result_file, paths.base)
    log.info(
        "NOTE_STYLE_PACK_BUILT sid=%s acc=%s pack=%s result=%s note_hash=%s prompt_salt=%s source_hash=%s char_len=%s word_len=%s truncated=%s model=%s fingerprint=%s",
        sid,
        account_id_str,
        pack_relative,
        result_relative,
        note_hash,
        prompt_salt,
        source_hash,
        char_len,
        word_len,
        note_truncated,
        _NOTE_STYLE_MODEL,
        fingerprint,
    )
    log_structured_event(
        "NOTE_STYLE_PACK_BUILT",
        logger=log,
        sid=sid,
        account_id=account_id_str,
        pack_path=pack_relative,
        result_path=result_relative,
        note_hash=note_hash,
        prompt_salt=prompt_salt,
        source_hash=source_hash,
        model=_NOTE_STYLE_MODEL,
        note_metrics=result_payload.get("note_metrics"),
        fingerprint_hash=fingerprint_hash,
    )

    entry = _serialize_entry(
        sid=sid,
        account_id=str(account_id),
        paths=paths,
        account_paths=account_paths,
        note_hash=note_hash,
        status="built",
        timestamp=timestamp,
    )

    totals = _update_index_for_account(
        sid=sid, paths=paths, account_id=str(account_id), entry=entry
    )
    _record_stage_progress(
        sid=sid, runs_root=runs_root_path, totals=totals, index_path=paths.index_file
    )

    return {
        "status": "completed",
        "packs_total": totals.get("total", 0),
        "packs_completed": totals.get("completed", 0),
        "prompt_salt": prompt_salt,
        "note_hash": note_hash,
    }


@dataclass
class _DebounceEntry:
    timer: threading.Timer


_DEBOUNCE_LOCK = threading.Lock()
_PENDING: dict[tuple[str, str], _DebounceEntry] = {}


def _schedule_timer(delay: float, fn: Callable[[], None]) -> threading.Timer:
    timer = threading.Timer(delay, fn)
    timer.daemon = True
    timer.start()
    return timer


def _debounce_delay_seconds() -> float:
    raw = os.getenv(_DEBOUNCE_MS_ENV)
    if raw is None:
        return _DEFAULT_DEBOUNCE_MS / 1000.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_DEBOUNCE_MS / 1000.0
    if value <= 0:
        return 0.0
    return value / 1000.0


def schedule_note_style_refresh(
    sid: str, account_id: str, *, runs_root: Path | str | None = None
) -> None:
    if not config.NOTE_STYLE_ENABLED:
        log.info(
            "NOTE_STYLE_DISABLED sid=%s account_id=%s", sid, account_id
        )
        return

    delay = _debounce_delay_seconds()

    def _run() -> None:
        try:
            build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
        except Exception:  # pragma: no cover - defensive
            log.exception(
                "NOTE_STYLE_BUILD_FAILED sid=%s account_id=%s", sid, account_id
            )
        finally:
            with _DEBOUNCE_LOCK:
                _PENDING.pop((sid, account_id), None)

    if delay <= 0:
        _run()
        return

    with _DEBOUNCE_LOCK:
        existing = _PENDING.pop((sid, account_id), None)
        if existing is not None:
            try:
                existing.timer.cancel()
            except Exception:
                pass
        timer = _schedule_timer(delay, _run)
        _PENDING[(sid, account_id)] = _DebounceEntry(timer=timer)


__all__ = [
    "NoteStyleResponseAccount",
    "discover_note_style_response_accounts",
    "build_note_style_pack_for_account",
    "schedule_note_style_refresh",
]

