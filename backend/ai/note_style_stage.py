"""Simplified builders for the note_style AI stage."""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, MutableMapping

from backend.ai.manifest import ensure_note_style_section
from backend.core.ai.paths import (
    NoteStylePaths,
    ensure_note_style_account_paths,
    ensure_note_style_paths,
    note_style_pack_filename,
    note_style_result_filename,
    normalize_note_style_account_id,
)
from backend.runflow.decider import record_stage


log = logging.getLogger(__name__)


_NOTE_STYLE_SYSTEM_PROMPT = (
    "You analyse customer notes and respond with structured JSON.\n"
    "Return exactly one JSON object using this schema:\n"
    '{"tone": string, "context_hints": {"timeframe": {"month": string|null, "relative": '
    'string|null}, "topic": string, "entities": {"creditor": string|null, "amount": '
    'number|null}}, "emphasis": [string], "confidence": number, "risk_flags": [string]}.\n'
    "Never include explanations or additional keys."
)

_NOTE_KEYS = {
    "note",
    "notes",
    "customer_note",
    "explain",
    "explanation",
}
_ZERO_WIDTH_TRANSLATION = {
    ord("\u200b"): " ",
    ord("\u200c"): " ",
    ord("\u200d"): " ",
    ord("\ufeff"): " ",
    ord("\u2060"): " ",
}

@dataclass(frozen=True)
class NoteStyleResponseAccount:
    """Details about a frontend response discovered for the stage."""

    account_id: str
    normalized_account_id: str
    response_path: Path
    response_relative: PurePosixPath
    pack_filename: str
    result_filename: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_root = os.getenv("RUNS_ROOT")
        return Path(env_root) if env_root else Path("runs")
    return Path(runs_root)


def _load_json_data(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("NOTE_STYLE_LOAD_JSON_FAILED path=%s", path, exc_info=True)
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("NOTE_STYLE_LOAD_JSON_INVALID path=%s", path, exc_info=True)
        return None


def _extract_note_text(payload: Any) -> str:
    if isinstance(payload, str):
        text = payload.strip()
        return text

    if isinstance(payload, Mapping):
        for key in _NOTE_KEYS:
            candidate = payload.get(key)
            text = _extract_note_text(candidate)
            if text:
                return text
        for value in payload.values():
            if isinstance(value, Mapping) or (
                isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray))
            ):
                text = _extract_note_text(value)
                if text:
                    return text
        return ""

    if isinstance(payload, Iterable) and not isinstance(payload, (bytes, bytearray)):
        for item in payload:
            text = _extract_note_text(item)
            if text:
                return text

    return ""


def _sanitize_note_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    translated = normalized.translate(_ZERO_WIDTH_TRANSLATION)
    collapsed = " ".join(translated.split())
    return collapsed.strip()


_BUREAU_KEYS = ("transunion", "experian", "equifax")

_BUREAU_CORE_FIELDS = (
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


def _clean_display_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_meta_name(
    meta_payload: Mapping[str, Any] | None, account_id: str
) -> str:
    if isinstance(meta_payload, Mapping):
        for key in ("heading_guess", "name"):
            candidate = _clean_display_text(meta_payload.get(key))
            if candidate:
                return candidate
    fallback = _clean_display_text(account_id)
    return fallback or str(account_id)


def _filter_bureau_fields(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    filtered: dict[str, Any] = {}
    for field in _BUREAU_CORE_FIELDS:
        value = payload.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            filtered[field] = text
        else:
            filtered[field] = value
    return filtered


def _build_majority_values(bureaus_payload: Mapping[str, Any]) -> dict[str, Any]:
    majority_payload = bureaus_payload.get("majority_values")
    majority_values = _filter_bureau_fields(majority_payload)
    if majority_values:
        return majority_values

    for bureau_key in _BUREAU_KEYS:
        bureau_values = _filter_bureau_fields(bureaus_payload.get(bureau_key))
        if bureau_values:
            return bureau_values
    return {}


def _extract_bureau_data(bureaus_payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(bureaus_payload, Mapping):
        return {}

    bureau_data: dict[str, Any] = {}
    majority_values = _build_majority_values(bureaus_payload)
    if majority_values:
        bureau_data["majority_values"] = majority_values

    per_bureau: dict[str, Any] = {}
    for bureau_key in _BUREAU_KEYS:
        filtered = _filter_bureau_fields(bureaus_payload.get(bureau_key))
        if filtered:
            per_bureau[bureau_key] = filtered
    if per_bureau:
        bureau_data["per_bureau"] = per_bureau

    return bureau_data


def _issue_type_from_entry(entry: Any) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    if entry.get("kind") != "issue":
        return None
    issue_type = _clean_display_text(entry.get("type"))
    return issue_type


def _extract_primary_issue_tag(tags_payload: Any) -> str | None:
    if isinstance(tags_payload, Mapping):
        issue_type = _issue_type_from_entry(tags_payload)
        if issue_type:
            return issue_type
        entries = tags_payload.get("tags")
        if isinstance(entries, Iterable) and not isinstance(entries, (str, bytes, bytearray)):
            for entry in entries:
                issue_type = _issue_type_from_entry(entry)
                if issue_type:
                    return issue_type
        return None

    if isinstance(tags_payload, Iterable) and not isinstance(tags_payload, (str, bytes, bytearray)):
        for entry in tags_payload:
            issue_type = _issue_type_from_entry(entry)
            if issue_type:
                return issue_type
    return None


def _resolve_response_path(sid: str, account_id: str, runs_root: Path) -> Path:
    return (runs_root / sid / "frontend" / "review" / "responses" / f"{account_id}.result.json").resolve()


def _coerce_path(value: Any) -> Path | None:
    if value is None:
        return None
    try:
        raw = os.fspath(value)
    except TypeError:
        return None

    text = str(raw).strip()
    if not text:
        return None

    try:
        return Path(text).resolve()
    except OSError:
        return Path(text)


def _lookup_manifest_account_entry(
    manifest_payload: Any, account_id: str
) -> Mapping[str, Any] | None:
    if not isinstance(manifest_payload, Mapping):
        return None

    artifacts = manifest_payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None

    cases_section = artifacts.get("cases")
    if not isinstance(cases_section, Mapping):
        return None

    accounts_section = cases_section.get("accounts")
    if not isinstance(accounts_section, Mapping):
        return None

    normalized = str(account_id).strip()
    raw_candidates: list[str] = []
    if normalized:
        raw_candidates.append(normalized)
        raw_candidates.append(normalized.lower())

    for piece in re.findall(r"(\d+)", normalized):
        trimmed = piece.lstrip("0") or "0"
        raw_candidates.append(trimmed)
        raw_candidates.append(trimmed.lower())

    candidates = [candidate for candidate in dict.fromkeys(raw_candidates) if candidate]

    for candidate in candidates:
        entry = accounts_section.get(candidate)
        if isinstance(entry, Mapping):
            return entry

        for key, value in accounts_section.items():
            if isinstance(key, str) and key.lower() == candidate.lower() and isinstance(value, Mapping):
                return value

    return None


def _resolve_account_context_paths(
    sid: str, account_id: str, runs_root: Path
) -> dict[str, Path]:
    run_dir = runs_root / sid
    manifest_payload = _load_json_data(run_dir / "manifest.json")
    account_entry = _lookup_manifest_account_entry(manifest_payload, account_id)

    resolved: dict[str, Path] = {}
    if isinstance(account_entry, Mapping):
        for key in ("dir", "meta", "bureaus", "tags"):
            candidate = _coerce_path(account_entry.get(key))
            if candidate is not None:
                resolved[key] = candidate

    account_dir = resolved.get("dir")
    if account_dir is None:
        account_dir = (run_dir / "cases" / "accounts" / account_id).resolve()
        resolved["dir"] = account_dir

    for key, filename in (("meta", "meta.json"), ("bureaus", "bureaus.json"), ("tags", "tags.json")):
        if key not in resolved:
            resolved[key] = (account_dir / filename).resolve()

    return resolved


def _relative_to_base(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _ensure_index_entry(
    *,
    paths: NoteStylePaths,
    account_id: str,
    pack_path: Path,
    result_path: Path,
    timestamp: str,
) -> Mapping[str, Any]:
    index_path = paths.index_file
    packs: list[MutableMapping[str, Any]] = []
    index_payload: MutableMapping[str, Any]

    existing = _load_json_data(index_path)
    if isinstance(existing, Mapping):
        existing_packs = existing.get("packs")
        if isinstance(existing_packs, list):
            for entry in existing_packs:
                if isinstance(entry, Mapping) and str(entry.get("account_id")) != account_id:
                    packs.append(dict(entry))

    packs.append(
        {
            "account_id": account_id,
            "status": "built",
            "pack_path": _relative_to_base(pack_path, paths.base),
            "result_path": _relative_to_base(result_path, paths.base),
            "built_at": timestamp,
        }
    )
    packs.sort(key=lambda entry: str(entry.get("account_id")))

    index_payload = {
        "version": 1,
        "updated_at": timestamp,
        "packs": packs,
        "totals": {"packs_total": len(packs)},
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index_payload


def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def discover_note_style_response_accounts(
    sid: str, *, runs_root: Path | str | None = None
) -> list[NoteStyleResponseAccount]:
    runs_root_path = _resolve_runs_root(runs_root)
    responses_dir = (runs_root_path / sid / "frontend" / "review" / "responses").resolve()

    if not responses_dir.is_dir():
        log.info("NOTE_STYLE_DISCOVERY sid=%s responses=%s usable=%s", sid, 0, 0)
        return []

    discovered: list[NoteStyleResponseAccount] = []
    total = 0
    usable = 0
    for candidate in sorted(responses_dir.glob("*.result.json"), key=lambda item: item.name):
        if not candidate.is_file():
            continue
        total += 1
        payload = _load_json_data(candidate)
        if not isinstance(payload, Mapping):
            continue
        note_text = _sanitize_note_text(_extract_note_text(payload))
        if not note_text:
            continue
        account_id = candidate.stem.replace(".result", "")
        normalized = normalize_note_style_account_id(account_id)
        pack_filename = note_style_pack_filename(account_id)
        result_filename = note_style_result_filename(account_id)
        relative = PurePosixPath(_relative_to_base(candidate, runs_root_path))
        discovered.append(
            NoteStyleResponseAccount(
                account_id=account_id,
                normalized_account_id=normalized,
                response_path=candidate.resolve(),
                response_relative=relative,
                pack_filename=pack_filename,
                result_filename=result_filename,
            )
        )
        usable += 1

    discovered.sort(key=lambda entry: entry.account_id)
    log.info(
        "NOTE_STYLE_DISCOVERY sid=%s responses=%s usable=%s",
        sid,
        total,
        usable,
    )
    return discovered


def _record_stage_snapshot(
    *,
    sid: str,
    runs_root: Path,
    index_payload: Mapping[str, Any],
) -> None:
    packs_snapshot = index_payload.get("packs")
    packs_total = len(packs_snapshot) if isinstance(packs_snapshot, list) else 0
    try:
        record_stage(
            sid,
            "note_style",
            status="built" if packs_total else "success",
            counts={"packs_total": packs_total},
            empty_ok=packs_total == 0,
            metrics={"packs_total": packs_total},
            runs_root=runs_root,
        )
    except Exception:  # pragma: no cover - defensive logging
        log.exception("NOTE_STYLE_STAGE_RECORD_FAILED sid=%s", sid)


def build_note_style_pack_for_account(
    sid: str, account_id: str, *, runs_root: Path | str | None = None
) -> Mapping[str, Any]:
    runs_root_path = _resolve_runs_root(runs_root)
    ensure_note_style_section(sid, runs_root=runs_root_path)
    paths = ensure_note_style_paths(runs_root_path, sid, create=True)

    response_path = _resolve_response_path(sid, account_id, runs_root_path)
    payload = _load_json_data(response_path)
    if not isinstance(payload, Mapping):
        log.info(
            "NOTE_STYLE_BUILD_SKIP sid=%s account_id=%s reason=no_response", sid, account_id
        )
        return {"status": "skipped", "reason": "no_response"}

    note_text = _sanitize_note_text(_extract_note_text(payload))
    if not note_text:
        log.info(
            "NOTE_STYLE_BUILD_SKIP sid=%s account_id=%s reason=no_note", sid, account_id
        )
        return {"status": "skipped", "reason": "no_note"}

    context_paths = _resolve_account_context_paths(sid, account_id, runs_root_path)
    meta_path = context_paths.get("meta")
    bureaus_path = context_paths.get("bureaus")
    tags_path = context_paths.get("tags")

    meta_candidate = _load_json_data(meta_path) if isinstance(meta_path, Path) else None
    meta_payload = meta_candidate if isinstance(meta_candidate, Mapping) else None

    bureaus_candidate = _load_json_data(bureaus_path) if isinstance(bureaus_path, Path) else None
    bureaus_payload = bureaus_candidate if isinstance(bureaus_candidate, Mapping) else None

    tags_payload = _load_json_data(tags_path) if isinstance(tags_path, Path) else None

    timestamp = _now_iso()
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    meta_name = _extract_meta_name(meta_payload, account_id)
    bureau_data = _extract_bureau_data(bureaus_payload)
    primary_issue_tag = _extract_primary_issue_tag(tags_payload)
    pack_context = {
        "meta_name": meta_name,
        "bureau_data": bureau_data,
        "primary_issue_tag": primary_issue_tag,
    }

    user_message_content: dict[str, Any] = {
        "note_text": note_text,
        "context": pack_context,
    }

    note_metrics = {
        "char_len": len(note_text),
        "word_len": len(note_text.split()),
    }
    pack_payload = {
        "sid": sid,
        "account_id": account_id,
        "model": "gpt-4o-mini",
        "built_at": timestamp,
        "note_text": note_text,
        "context": pack_context,
        "note_metrics": note_metrics,
        "messages": [
            {"role": "system", "content": _NOTE_STYLE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message_content},
        ],
    }
    _write_jsonl(account_paths.pack_file, pack_payload)
    if account_paths.result_file.exists():
        try:
            account_paths.result_file.unlink()
        except OSError:
            log.warning(
                "NOTE_STYLE_RESULT_CLEANUP_FAILED sid=%s account_id=%s path=%s",
                sid,
                account_id,
                account_paths.result_file,
                exc_info=True,
            )
    if account_paths.debug_file.exists():
        try:
            account_paths.debug_file.unlink()
        except OSError:
            log.warning(
                "NOTE_STYLE_DEBUG_CLEANUP_FAILED sid=%s account_id=%s path=%s",
                sid,
                account_id,
                account_paths.debug_file,
                exc_info=True,
            )
    index_payload = _ensure_index_entry(
        paths=paths,
        account_id=account_id,
        pack_path=account_paths.pack_file,
        result_path=account_paths.result_file,
        timestamp=timestamp,
    )
    _record_stage_snapshot(sid=sid, runs_root=runs_root_path, index_payload=index_payload)

    log.info(
        "NOTE_STYLE_PACK_BUILT sid=%s account_id=%s pack=%s",
        sid,
        account_id,
        account_paths.pack_file,
    )

    return {
        "status": "completed",
        "packs_total": index_payload.get("totals", {}).get("packs_total", 1),
        "note_metrics": note_metrics,
    }


def schedule_note_style_refresh(
    sid: str, account_id: str, *, runs_root: Path | str | None = None
) -> None:
    if not config.NOTE_STYLE_ENABLED:
        log.info("NOTE_STYLE_DISABLED sid=%s account_id=%s", sid, account_id)
        return
    try:
        build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    except Exception:  # pragma: no cover - defensive logging
        log.exception("NOTE_STYLE_REFRESH_FAILED sid=%s account_id=%s", sid, account_id)


__all__ = [
    "NoteStyleResponseAccount",
    "discover_note_style_response_accounts",
    "build_note_style_pack_for_account",
    "schedule_note_style_refresh",
]
