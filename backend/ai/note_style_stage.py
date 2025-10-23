"""Build note_style AI stage artifacts from frontend review responses."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from backend.core.ai.paths import (
    NoteStyleAccountPaths,
    NoteStylePaths,
    ensure_note_style_account_paths,
    ensure_note_style_paths,
)
from backend.core.io.json_io import _atomic_write_json
from backend.runflow.decider import record_stage


log = logging.getLogger(__name__)

_INDEX_SCHEMA_VERSION = 1
_PROMPT_PEPPER_ENV = "NOTE_STYLE_PROMPT_PEPPER"
_DEBOUNCE_MS_ENV = "NOTE_STYLE_DEBOUNCE_MS"
_DEFAULT_DEBOUNCE_MS = 750

_DEFAULT_PEPPER = "finance-note-style"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        return Path(env_value) if env_value else Path("runs")
    return Path(runs_root)


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


def _write_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    serialized = json.dumps(row, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(serialized + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


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


def _pepper_bytes() -> bytes:
    value = os.getenv(_PROMPT_PEPPER_ENV)
    text = value if value else _DEFAULT_PEPPER
    return text.encode("utf-8")


def _source_hash(text: str) -> str:
    normalized = " ".join(text.split()).strip().lower()
    digest = hashlib.sha256()
    digest.update(normalized.encode("utf-8"))
    return digest.hexdigest()


def _prompt_salt(sid: str, account_id: str, source_hash: str) -> str:
    message = f"{sid}:{account_id}:{source_hash}".encode("utf-8")
    return hmac.new(_pepper_bytes(), message, hashlib.sha256).hexdigest()[:16]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _tokens(note: str) -> set[str]:
    cleaned = note.replace("/", " ").replace("-", " ")
    return {token for token in cleaned.lower().split() if token}


def _contains_phrase(text: str, *phrases: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _tone_features(note: str) -> tuple[str, float, list[str]]:
    text = note.lower()
    risk_flags: list[str] = []

    if any(word in text for word in ("angry", "frustrated", "furious", "upset")):
        tone = "frustrated"
        confidence = 0.85
        risk_flags.append("escalation_risk")
    elif any(word in text for word in ("urgent", "immediately", "asap", "right away")):
        tone = "urgent"
        confidence = 0.8
    elif any(word in text for word in ("sorry", "apologize", "apologies")):
        tone = "apologetic"
        confidence = 0.75
    elif any(word in text for word in ("please", "thank", "appreciate", "grateful")):
        tone = "conciliatory"
        confidence = 0.7
    else:
        tone = "neutral"
        confidence = 0.45

    if any(word in text for word in ("lawsuit", "sue", "court", "legal action")):
        risk_flags.append("legal_threat")

    return tone, confidence, _unique(risk_flags)


def _context_hints(note: str, words: set[str]) -> tuple[list[str], float, list[str]]:
    lowered = note.lower()
    hints: list[str] = []
    risk_flags: list[str] = []

    if _contains_phrase(lowered, "identity theft", "identity fraud"):
        hints.append("identity_theft")
        risk_flags.append("identity_theft_claim")
    if "fraud" in words or "scam" in words:
        hints.append("fraud_reported")
        risk_flags.append("fraud_claim")
    if _contains_phrase(lowered, "bank error", "bank mistake", "their error") or (
        "bank" in words and "error" in words
    ):
        hints.append("lender_fault")
    if _contains_phrase(lowered, "paid in full", "already paid") or (
        "paid" in words and "proof" in words
    ):
        hints.append("payment_dispute")
    if any(word in words for word in ("hardship", "unemployed", "layoff", "laid")):
        hints.append("financial_hardship")
    if any(word in words for word in ("military", "deployment", "service")):
        hints.append("military_service")
    if any(word in words for word in ("medical", "hospital", "surgery", "illness")):
        hints.append("medical_event")
    if any(word in words for word in ("divorce", "separated", "widow", "widowed")):
        hints.append("family_change")

    confidence = 0.4 if not hints else min(0.9, 0.55 + 0.1 * len(hints))
    return _unique(hints), confidence, _unique(risk_flags)


def _emphasis(note: str, words: set[str]) -> tuple[list[str], float, list[str]]:
    lowered = note.lower()
    emphasis: list[str] = []
    risk_flags: list[str] = []

    if any(word in words for word in ("dispute", "investigate", "challenge")):
        emphasis.append("dispute_resolution")
    if any(word in words for word in ("remove", "delete", "correct", "update", "fix")):
        emphasis.append("correct_record")
    if _contains_phrase(lowered, "please help") or "assistance" in words:
        emphasis.append("support_request")
    if _contains_phrase(lowered, "bank error") or (
        "bank" in words and "error" in words
    ):
        emphasis.append("lender_error")
    if _contains_phrase(lowered, "identity theft"):
        emphasis.append("protect_identity")
    if _contains_phrase(lowered, "already paid") or "receipt" in words:
        emphasis.append("confirm_payment")

    confidence = 0.4 if not emphasis else min(0.9, 0.6 + 0.08 * len(emphasis))
    if "legal" in words or _contains_phrase(lowered, "legal action"):
        risk_flags.append("legal_language")
    return _unique(emphasis), confidence, _unique(risk_flags)


def _extract_features(note: str) -> dict[str, Any]:
    tokens = _tokens(note)
    tone_value, tone_confidence, tone_risks = _tone_features(note)
    hints, hints_confidence, hints_risks = _context_hints(note, tokens)
    emphasis_values, emphasis_confidence, emphasis_risks = _emphasis(note, tokens)

    return {
        "tone": {
            "value": tone_value,
            "confidence": round(tone_confidence, 3),
            "risk_flags": tone_risks,
        },
        "context_hints": {
            "values": hints,
            "confidence": round(hints_confidence, 3),
            "risk_flags": hints_risks,
        },
        "emphasis": {
            "values": emphasis_values,
            "confidence": round(emphasis_confidence, 3),
            "risk_flags": emphasis_risks,
        },
    }


def _load_response_note(response_path: Path) -> tuple[str, Mapping[str, Any] | None]:
    try:
        raw = response_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return "", None
    except OSError:
        log.warning("NOTE_STYLE_RESPONSE_READ_FAILED path=%s", response_path, exc_info=True)
        return "", None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("NOTE_STYLE_RESPONSE_INVALID_JSON path=%s", response_path, exc_info=True)
        return "", None

    if not isinstance(payload, Mapping):
        return "", None

    answers = payload.get("answers")
    if not isinstance(answers, Mapping):
        return "", payload

    explanation = _normalize_text(answers.get("explanation"))
    return explanation, payload


def _index_items(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    items = payload.get("items")
    if isinstance(items, Sequence):
        return [dict(entry) for entry in items if isinstance(entry, Mapping)]
    packs = payload.get("packs")
    if isinstance(packs, Sequence):
        return [dict(entry) for entry in packs if isinstance(entry, Mapping)]
    return []


def _serialize_entry(
    *,
    sid: str,
    account_id: str,
    paths: NoteStylePaths,
    account_paths: NoteStyleAccountPaths,
    source_hash: str,
    features: Mapping[str, Any],
    prompt_salt: str,
    status: str,
    timestamp: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "account_id": account_id,
        "sid": sid,
        "pack": _relativize(account_paths.pack_file, paths.base),
        "result": _relativize(account_paths.result_file, paths.base),
        "status": status,
        "source_hash": source_hash,
        "prompt_salt": prompt_salt,
        "built_at": timestamp,
        "updated_at": timestamp,
        "completed_at": timestamp,
    }

    for key in ("tone", "context_hints", "emphasis"):
        value = features.get(key)
        if isinstance(value, Mapping):
            payload[key] = dict(value)

    return payload


def _compute_totals(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    total = 0
    completed = 0
    failed = 0
    for entry in items:
        status = _normalize_text(entry.get("status")).lower()
        if status in {"", "skipped"}:
            continue
        total += 1
        if status in {"completed", "success"}:
            completed += 1
        elif status in {"failed", "error"}:
            failed += 1
    if total == 0 and completed == 0 and failed == 0:
        return {"total": 0, "completed": 0, "failed": 0}
    return {"total": total, "completed": completed, "failed": failed}


def _write_index(
    *,
    sid: str,
    paths: NoteStylePaths,
    items: Sequence[Mapping[str, Any]],
) -> None:
    timestamp = _now_iso()
    totals = _compute_totals(items)
    document = {
        "schema_version": _INDEX_SCHEMA_VERSION,
        "sid": sid,
        "generated_at": timestamp,
        "packs_dir": _relativize(paths.packs_dir, paths.base),
        "results_dir": _relativize(paths.results_dir, paths.base),
        "items": list(items),
        "totals": totals,
    }
    _atomic_write_json(paths.index_file, document)


def _remove_account_artifacts(account_paths: NoteStyleAccountPaths) -> None:
    for path in (account_paths.pack_file, account_paths.result_file):
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
    existing = _load_json_mapping(paths.index_file)
    items = _index_items(existing)

    normalized_account = str(account_id)
    rewritten: list[dict[str, Any]] = []
    replaced = False
    for item in items:
        if str(item.get("account_id")) == normalized_account:
            if entry is not None:
                rewritten.append(dict(entry))
                replaced = True
            continue
        rewritten.append(item)

    if not replaced and entry is not None:
        rewritten.append(dict(entry))

    rewritten.sort(key=lambda item: str(item.get("account_id", "")))
    _write_index(sid=sid, paths=paths, items=rewritten)
    return _compute_totals(rewritten)


def _record_stage_progress(
    *, sid: str, runs_root: Path, totals: Mapping[str, int], index_path: Path
) -> None:
    packs_total = int(totals.get("total", 0))
    packs_completed = int(totals.get("completed", 0))
    packs_failed = int(totals.get("failed", 0))

    if packs_total == packs_completed:
        status: str = "success"
    else:
        status = "built" if packs_failed == 0 else "error"

    empty_ok = packs_total == 0
    if empty_ok:
        status = "success"

    counts = {
        "packs_total": packs_total,
        "packs_completed": packs_completed,
        "packs_failed": packs_failed,
    }
    metrics = {
        "packs_total": packs_total,
        "packs_completed": packs_completed,
    }
    results = {"index_path": str(index_path)}

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
    run_dir = runs_root_path / sid
    response_path = (
        run_dir
        / "frontend"
        / "review"
        / "responses"
        / f"{account_id}.result.json"
    )

    note_text, response_payload = _load_response_note(response_path)
    paths = ensure_note_style_paths(runs_root_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    if not note_text:
        totals = _update_index_for_account(sid=sid, paths=paths, account_id=account_id, entry=None)
        _remove_account_artifacts(account_paths)
        _record_stage_progress(
            sid=sid, runs_root=runs_root_path, totals=totals, index_path=paths.index_file
        )
        return {
            "status": "skipped",
            "reason": "empty_note",
            "packs_total": totals.get("total", 0),
            "packs_completed": totals.get("completed", 0),
        }

    features = _extract_features(note_text)
    source_hash = _source_hash(note_text)
    prompt_salt = _prompt_salt(sid, str(account_id), source_hash)
    timestamp = _now_iso()

    existing_index = _load_json_mapping(paths.index_file)
    items = _index_items(existing_index)
    for item in items:
        if str(item.get("account_id")) == str(account_id):
            if item.get("source_hash") == source_hash:
                totals = _compute_totals(items)
                _record_stage_progress(
                    sid=sid,
                    runs_root=runs_root_path,
                    totals=totals,
                    index_path=paths.index_file,
                )
                return {
                    "status": "unchanged",
                    "packs_total": totals.get("total", 0),
                    "packs_completed": totals.get("completed", 0),
                }

    pack_payload = {
        "sid": sid,
        "account_id": str(account_id),
        "prompt_salt": prompt_salt,
        "source_hash": source_hash,
        "features": features,
        "built_at": timestamp,
    }
    result_payload = {
        "sid": sid,
        "account_id": str(account_id),
        "analysis": features,
        "prompt_salt": prompt_salt,
        "source_hash": source_hash,
        "evaluated_at": timestamp,
    }

    _write_jsonl(account_paths.pack_file, pack_payload)
    _write_jsonl(account_paths.result_file, result_payload)

    entry = _serialize_entry(
        sid=sid,
        account_id=str(account_id),
        paths=paths,
        account_paths=account_paths,
        source_hash=source_hash,
        features=features,
        prompt_salt=prompt_salt,
        status="completed",
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
    "build_note_style_pack_for_account",
    "schedule_note_style_refresh",
]
