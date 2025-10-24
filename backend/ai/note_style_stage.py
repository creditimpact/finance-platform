"""Simplified builders for the note_style AI stage."""

from __future__ import annotations

import json
import logging
import os
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, MutableMapping

from backend import config
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
    "Return exactly one JSON object using this schema:"
    " {\"tone\": string, \"context_hints\": {\"timeframe\": {\"month\": string|null,"
    " \"relative\": string|null}, \"topic\": string, \"entities\": {\"creditor\": string|null,"
    " \"amount\": number|null}}, \"emphasis\": [string], \"confidence\": number,"
    " \"risk_flags\": [string] }.\n"
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


def _resolve_response_path(sid: str, account_id: str, runs_root: Path) -> Path:
    return (runs_root / sid / "frontend" / "review" / "responses" / f"{account_id}.result.json").resolve()


def _resolve_account_dir(sid: str, account_id: str, runs_root: Path) -> Path:
    return (runs_root / sid / "cases" / "accounts" / account_id).resolve()


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


def _primary_issue_tag(tags_payload: Any) -> str | None:
    if isinstance(tags_payload, Mapping):
        entries = tags_payload.get("tags")
    else:
        entries = tags_payload

    if isinstance(entries, Iterable) and not isinstance(entries, (str, bytes, bytearray)):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            kind = str(entry.get("kind") or "").strip().lower()
            if kind != "issue":
                continue
            tag = str(entry.get("type") or entry.get("tag") or "").strip()
            if tag:
                return tag
    return None


def _resolve_account_name(meta_payload: Mapping[str, Any] | None, account_id: str) -> str:
    if isinstance(meta_payload, Mapping):
        for key in ("heading_guess", "display_name", "name", "account_name", "creditor_name"):
            value = meta_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return account_id


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

    account_dir = _resolve_account_dir(sid, account_id, runs_root_path)
    meta_candidate = _load_json_data(account_dir / "meta.json")
    meta_payload = meta_candidate if isinstance(meta_candidate, Mapping) else None
    bureaus_candidate = _load_json_data(account_dir / "bureaus.json")
    bureaus_payload = bureaus_candidate if isinstance(bureaus_candidate, Mapping) else None
    tags_payload = _load_json_data(account_dir / "tags.json")

    timestamp = _now_iso()
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    primary_issue = _primary_issue_tag(tags_payload)
    account_name = _resolve_account_name(meta_payload, account_id)

    pack_context = {
        "meta_name": account_name,
        "primary_issue_tag": primary_issue,
        "bureau_data": bureaus_payload or {},
        "note_text": note_text,
    }

    debug_snapshot = {
        "sid": sid,
        "account_id": account_id,
        "collected_at": timestamp,
        "meta": meta_payload,
        "bureaus": bureaus_payload,
        "tags": tags_payload,
        "note_text": note_text,
    }

    pack_payload = {
        "sid": sid,
        "account_id": account_id,
        "model": config.NOTE_STYLE_MODEL,
        "built_at": timestamp,
        "context": pack_context,
        "messages": [
            {"role": "system", "content": _NOTE_STYLE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(pack_context, ensure_ascii=False)},
        ],
    }

    note_metrics = {
        "char_len": len(note_text),
        "word_len": len(note_text.split()),
    }
    result_payload = {
        "sid": sid,
        "account_id": account_id,
        "analysis": None,
        "note_metrics": note_metrics,
    }

    _write_jsonl(account_paths.pack_file, pack_payload)
    _write_jsonl(account_paths.result_file, result_payload)
    account_paths.debug_file.write_text(
        json.dumps(debug_snapshot, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
