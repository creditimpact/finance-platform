"""Frontend pack generation helpers."""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.core.io.json_io import _atomic_write_json as _shared_atomic_write_json
from backend.core.paths.frontend_review import (
    ensure_frontend_review_dirs,
    get_frontend_review_paths,
)
from backend.core.runflow import (
    record_frontend_responses_progress,
    runflow_account_steps_enabled,
    runflow_step,
)
from backend.core.runflow.io import (
    compose_hint,
    format_exception_tail,
    runflow_stage_end,
    runflow_stage_error,
    runflow_stage_start,
)
from backend.frontend.packs.config import (
    FrontendStageConfig,
    load_frontend_stage_config,
)
from backend.domain.claims import CLAIM_FIELD_LINK_MAP

log = logging.getLogger(__name__)

_BUREAU_BADGES: Mapping[str, Mapping[str, str]] = {
    "transunion": {"id": "transunion", "label": "TransUnion", "short_label": "TU"},
    "equifax": {"id": "equifax", "label": "Equifax", "short_label": "EF"},
    "experian": {"id": "experian", "label": "Experian", "short_label": "EX"},
}

_BUREAU_ORDER: tuple[str, ...] = ("transunion", "experian", "equifax")

_DISPLAY_SCHEMA_VERSION = "1.2"

_STAGE_PAYLOAD_MODE_MINIMAL = "minimal"
_STAGE_PAYLOAD_MODE_FULL = "full"
_STAGE_PAYLOAD_MODES: set[str] = {_STAGE_PAYLOAD_MODE_MINIMAL, _STAGE_PAYLOAD_MODE_FULL}


_QUESTION_SET = [
    {"id": "ownership", "prompt": "Do you own this account?"},
    {"id": "recognize", "prompt": "Do you recognize this account on your report?"},
    {"id": "explanation", "prompt": "Anything else we should know about this account?"},
    {"id": "identity_theft", "prompt": "Is this account tied to identity theft?"},
]


def _env_flag_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"", "0", "false", "no", "off"}:
        return False
    return True


def _claim_field_links_payload() -> dict[str, list[str]]:
    return {key: list(values) for key, values in CLAIM_FIELD_LINK_MAP.items()}


def _coerce_question_list(questions: Any) -> list[dict[str, Any]]:
    if not isinstance(questions, Sequence) or isinstance(
        questions, (str, bytes, bytearray)
    ):
        return []

    normalized: list[dict[str, Any]] = []
    for question in questions:
        if isinstance(question, Mapping):
            normalized.append(dict(question))

    return normalized


def _resolve_stage_pack_questions(
    *,
    existing_pack: Mapping[str, Any] | None,
    question_set: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if isinstance(existing_pack, Mapping):
        existing_questions = _coerce_question_list(existing_pack.get("questions"))
        if existing_questions:
            return existing_questions

    if question_set is None:
        return []

    return _coerce_question_list(question_set)


_POINTER_KEYS: tuple[str, ...] = (
    "meta",
    "tags",
    "raw",
    "bureaus",
    "flat",
    "summary",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _default_runs_root() -> Path:
    root_env = os.getenv("RUNS_ROOT")
    return Path(root_env) if root_env else Path("runs")


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        return _default_runs_root()
    return Path(runs_root)


def _frontend_packs_enabled() -> bool:
    value = os.getenv("ENABLE_FRONTEND_PACKS", "1")
    return value not in {"0", "false", "False"}


def _frontend_packs_lean_enabled() -> bool:
    value = os.getenv("FRONTEND_PACKS_LEAN", "1")
    return value not in {"0", "false", "False"}


def _frontend_packs_debug_mirror_enabled() -> bool:
    value = os.getenv("FRONTEND_PACKS_DEBUG_MIRROR", "0")
    return value not in {"0", "false", "False"}


def _resolve_stage_payload_mode() -> str:
    value = os.getenv("FRONTEND_STAGE_PAYLOAD", _STAGE_PAYLOAD_MODE_MINIMAL)
    if not value:
        return _STAGE_PAYLOAD_MODE_MINIMAL

    normalized = value.strip().lower()
    if normalized not in _STAGE_PAYLOAD_MODES:
        log.warning(
            "FRONTEND_STAGE_PAYLOAD_INVALID value=%s", value,
        )
        return _STAGE_PAYLOAD_MODE_MINIMAL

    return normalized


def _frontend_review_create_empty_index_enabled() -> bool:
    return _env_flag_enabled("FRONTEND_REVIEW_CREATE_EMPTY_INDEX", False)


def _log_stage_paths(
    sid: str,
    config: FrontendStageConfig,
    canonical_paths: Mapping[str, str],
) -> None:
    base_path = canonical_paths.get("frontend_base") or config.stage_dir.parent
    log.info(
        "FRONTEND_REVIEW_PATHS sid=%s base=%s dir=%s packs=%s results=%s",
        sid,
        str(base_path),
        str(config.stage_dir),
        str(config.packs_dir),
        str(config.responses_dir),
    )


def _frontend_build_lock_path(run_dir: Path) -> Path:
    return run_dir / "frontend" / ".locks" / "build.lock"


def _acquire_frontend_build_lock(run_dir: Path, sid: str) -> tuple[str, Path | None]:
    lock_path = _frontend_build_lock_path(run_dir)

    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "FRONTEND_BUILD_LOCK_DIR_FAILED sid=%s path=%s",
            sid,
            lock_path.parent,
            exc_info=True,
        )

    try:
        fd = os.open(os.fspath(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return "locked", lock_path
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "FRONTEND_BUILD_LOCK_ACQUIRE_FAILED sid=%s path=%s",
            sid,
            lock_path,
            exc_info=True,
        )
        return "error", None

    payload = {
        "sid": sid,
        "acquired_at": time.time(),
        "pid": os.getpid(),
    }

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "FRONTEND_BUILD_LOCK_WRITE_FAILED sid=%s path=%s",
            sid,
            lock_path,
            exc_info=True,
        )
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "FRONTEND_BUILD_LOCK_CLEANUP_FAILED sid=%s path=%s",
                sid,
                lock_path,
                exc_info=True,
            )
        return "error", None

    return "acquired", lock_path


def _release_frontend_build_lock(lock_path: Path, sid: str) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "FRONTEND_BUILD_LOCK_RELEASE_FAILED sid=%s path=%s",
            sid,
            lock_path,
            exc_info=True,
        )


def _log_build_summary(
    sid: str,
    *,
    packs_count: int,
    last_built_at: str | None,
) -> None:
    log.info(
        "FRONTEND_REVIEW_BUILD_COMPLETE sid=%s packs_count=%s last_built_at=%s",
        sid,
        packs_count,
        last_built_at or "-",
    )


def _count_frontend_responses(responses_dir: Path) -> int:
    if not responses_dir.is_dir():
        return 0

    total = 0
    for entry in responses_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.name.endswith(".tmp"):
            continue
        total += 1
    return total


def _emit_responses_scan(_sid: str, responses_dir: Path) -> int:
    return _count_frontend_responses(responses_dir)


def _account_sort_key(path: Path) -> tuple[int, Any]:
    name = path.name
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:  # pragma: no cover - defensive logging
        log.warning("FRONTEND_PACK_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("FRONTEND_PACK_PARSE_FAILED path=%s", path, exc_info=True)
        return None

    if not isinstance(payload, Mapping):
        return None
    return payload


def _load_json_payload(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:  # pragma: no cover - defensive logging
        log.warning("FRONTEND_PACK_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("FRONTEND_PACK_PARSE_FAILED path=%s", path, exc_info=True)
        return None


def _is_frontend_review_index(path: Path) -> bool:
    normalized = path.as_posix()
    return normalized.endswith("frontend/review/index.json")


def _atomic_write_frontend_review_index(path: Path, payload: Any) -> None:
    directory = path.parent
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="index.", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        shutil.move(tmp_path, os.fspath(path))
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_write_json(path: Path | str, payload: Any) -> None:
    path_obj = Path(path)
    if _is_frontend_review_index(path_obj):
        _atomic_write_frontend_review_index(path_obj, payload)
        return
    _shared_atomic_write_json(path_obj, payload)


def _write_json_if_changed(path: Path, payload: Any) -> bool:
    current = _load_json_payload(path)
    if current == payload:
        return False

    _atomic_write_json(path, payload)
    return True


def _ensure_frontend_index_redirect_stub(path: Path, *, force: bool = False) -> None:
    """Write the legacy ``frontend/index.json`` redirect if it is missing."""

    if path.exists() and not force:
        return

    # Backward-compatibility stub for clients still reading the legacy path.
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, {"redirect": "frontend/review/index.json"})


def _relative_to_run_dir(path: Path, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _relative_to_stage_dir(path: Path, stage_dir: Path) -> str:
    try:
        return path.relative_to(stage_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return None


def _safe_sha1(path: Path) -> str | None:
    try:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:  # pragma: no cover - defensive logging
        log.warning("FRONTEND_PACK_SHA1_FAILED path=%s", path, exc_info=True)
        return None


def _resolve_pack_output_path(pack_path: str, run_dir: Path) -> Path:
    candidate = Path(pack_path)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate


def _pack_requires_pointer_backfill(payload: Mapping[str, Any]) -> bool:
    pointers = payload.get("pointers") if isinstance(payload, Mapping) else None
    if pointers is None:
        return False
    if not isinstance(pointers, Mapping):
        return True

    for key in _POINTER_KEYS:
        value = pointers.get(key)
        if not isinstance(value, str) or not value:
            return True

    return False


def _index_requires_pointer_backfill(
    index_payload: Mapping[str, Any], run_dir: Path
) -> bool:
    candidates: list[Mapping[str, Any]] = []

    accounts = index_payload.get("accounts")
    if isinstance(accounts, Sequence):
        for entry in accounts:
            if isinstance(entry, Mapping):
                candidates.append(entry)

    packs = index_payload.get("packs")
    if isinstance(packs, Sequence):
        for entry in packs:
            if isinstance(entry, Mapping):
                candidates.append(entry)

    if not candidates:
        return False

    seen_paths: set[str] = set()
    for entry in candidates:
        pack_path_value = entry.get("pack_path")
        if not isinstance(pack_path_value, str):
            pack_path_value = entry.get("path") if isinstance(entry, Mapping) else None
        if not isinstance(pack_path_value, str):
            continue
        if pack_path_value in seen_paths:
            continue
        seen_paths.add(pack_path_value)
        if "frontend/accounts/" in pack_path_value:
            return True
        pack_path = _resolve_pack_output_path(pack_path_value, run_dir)
        pack_payload = _load_json_payload(pack_path)
        if not isinstance(pack_payload, Mapping):
            return True
        if _pack_requires_pointer_backfill(pack_payload):
            return True

    return False


def _log_done(sid: str, packs: int, **extras: Any) -> None:
    details = [f"sid={sid}", f"packs={packs}"]
    for key, value in sorted(extras.items()):
        if value is None:
            continue
        details.append(f"{key}={value}")
    log.info("PACKGEN_FRONTEND_DONE %s", " ".join(details))


def _extract_text(value: Any) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    if isinstance(value, Mapping):
        for key in (
            "text",
            "label",
            "display",
            "value",
            "normalized",
            "name",
        ):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _extract_summary_labels(summary: Mapping[str, Any]) -> Mapping[str, str | None]:
    labels = summary.get("labels")
    creditor = None
    account_type = None
    status = None

    if isinstance(labels, Mapping):
        creditor = _extract_text(labels.get("creditor") or labels.get("creditor_name"))
        account_type = _extract_text(labels.get("account_type"))
        status = _extract_text(labels.get("status") or labels.get("account_status"))

    if creditor is None:
        creditor = _extract_text(summary.get("creditor") or summary.get("creditor_name"))

    normalized = summary.get("normalized")
    if isinstance(normalized, Mapping):
        account_type = account_type or _extract_text(normalized.get("account_type"))
        status = status or _extract_text(normalized.get("status") or normalized.get("account_status"))

    return {
        "creditor_name": creditor,
        "account_type": account_type,
        "status": status,
    }


def _extract_last4(displays: Iterable[str]) -> Mapping[str, str | None]:
    digits: list[str] = []
    cleaned_display = None
    for display in displays:
        if not display:
            continue
        trimmed = str(display).strip()
        if not trimmed:
            continue
        cleaned_display = cleaned_display or trimmed
        numbers = re.sub(r"\D", "", trimmed)
        if len(numbers) >= 4:
            digits.append(numbers[-4:])

    last4_value = None
    if digits:
        # Prefer the most common last4
        seen: dict[str, int] = {}
        for candidate in digits:
            seen[candidate] = seen.get(candidate, 0) + 1
        last4_value = max(seen.items(), key=lambda item: (item[1], item[0]))[0]

    return {"display": cleaned_display, "last4": last4_value}


def _derive_masked_display(last4_payload: Mapping[str, Any] | None) -> str:
    """Return a masked display string derived from the last-4 payload."""

    display_value: str | None = None
    last4_digits: str | None = None

    if isinstance(last4_payload, Mapping):
        raw_display = last4_payload.get("display")
        if isinstance(raw_display, str):
            display_value = raw_display.strip() or None
        elif raw_display is not None:
            display_value = str(raw_display).strip() or None

        raw_last4 = last4_payload.get("last4")
        if isinstance(raw_last4, str):
            cleaned = re.sub(r"\D", "", raw_last4)
            last4_digits = cleaned[-4:] if cleaned else None
        elif raw_last4 is not None:
            cleaned = re.sub(r"\D", "", str(raw_last4))
            last4_digits = cleaned[-4:] if cleaned else None

    if display_value:
        return display_value

    if last4_digits:
        return f"****{last4_digits}"

    return "****"


def _coerce_display_text(value: Any) -> str:
    """Normalize optional display text into a stable string value."""

    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else ""

    if value is None:
        return ""

    cleaned = str(value).strip()
    return cleaned if cleaned else ""


def _collect_field_per_bureau(
    bureaus: Mapping[str, Mapping[str, Any]], field: str
) -> tuple[dict[str, str], str | None]:
    values: dict[str, str] = {}
    unique: set[str] = set()
    for bureau, payload in bureaus.items():
        value = payload.get(field)
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                values[bureau] = trimmed
                unique.add(trimmed)
        elif value is not None:
            coerced = str(value)
            if coerced.strip():
                values[bureau] = coerced.strip()
                unique.add(coerced.strip())

    consensus = unique.pop() if len(unique) == 1 else None
    return values, consensus


def _collect_field_text_values(
    bureaus: Mapping[str, Mapping[str, Any]], field: str
) -> dict[str, str]:
    values: dict[str, str] = {}
    for bureau in _BUREAU_ORDER:
        payload = bureaus.get(bureau)
        if not isinstance(payload, Mapping):
            continue
        raw_value = payload.get(field)
        if isinstance(raw_value, str):
            cleaned = raw_value.strip()
        elif raw_value is not None:
            cleaned = str(raw_value).strip()
        else:
            cleaned = ""
        if cleaned:
            values[bureau] = cleaned
    return values


def _resolve_account_number_consensus(per_bureau: Mapping[str, str]) -> str:
    duplicates = set()
    ordered_values: list[str] = []
    for bureau in _BUREAU_ORDER:
        value = per_bureau.get(bureau)
        if value and value != "--":
            ordered_values.append(value)
    if not ordered_values:
        return "--"
    counts = Counter(ordered_values)
    duplicates = {value for value, count in counts.items() if count >= 2}
    if duplicates:
        for bureau in _BUREAU_ORDER:
            value = per_bureau.get(bureau)
            if value in duplicates:
                return value
    for bureau in _BUREAU_ORDER:
        value = per_bureau.get(bureau)
        if value and value != "--":
            return value
    return "--"


def _resolve_majority_consensus(per_bureau: Mapping[str, str]) -> str:
    values: list[str] = []
    for bureau in _BUREAU_ORDER:
        value = per_bureau.get(bureau)
        if value and value != "--":
            values.append(value)
    if not values:
        return "--"
    counts = Counter(values)
    majority_values = {value for value, count in counts.items() if count >= 2}
    if majority_values:
        for bureau in _BUREAU_ORDER:
            value = per_bureau.get(bureau)
            if value in majority_values:
                return value
    for bureau in _BUREAU_ORDER:
        value = per_bureau.get(bureau)
        if value and value != "--":
            return value
    return "--"


def _load_raw_lines(path: Path) -> Sequence[str]:
    payload = _load_json_payload(path)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        lines: list[str] = []
        for entry in payload:
            if isinstance(entry, Mapping):
                text = entry.get("text")
                if isinstance(text, str):
                    lines.append(text)
            elif isinstance(entry, str):
                lines.append(entry)
        return lines
    return []


def holder_name_from_raw_lines(raw_lines: list[str]) -> str | None:
    preferred: list[str] = []
    fallback: list[str] = []
    for candidate in raw_lines:
        if not isinstance(candidate, str):
            continue
        stripped = candidate.strip()
        if not stripped:
            continue
        if not _looks_like_holder_heading(stripped):
            continue

        if re.search(r"[ /]", stripped):
            preferred.append(stripped)
        else:
            fallback.append(stripped)

    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return None


def _looks_like_holder_heading(text: str) -> bool:
    if not text:
        return False

    if text != text.upper():
        return False

    letters = sum(1 for char in text if char.isalpha())
    if letters < 2:
        return False

    digits = sum(1 for char in text if char.isdigit())
    if digits > max(2, len(text) // 5):
        return False

    if re.search(r"\b(ACCOUNT|BALANCE|PAYMENT|VERIFIED|OPENED|REPORTED)\b", text):
        return False

    if not re.fullmatch(r"[A-Z0-9 &'./-]+", text):
        return False

    return True


def _derive_holder_name_from_summary(
    summary: Mapping[str, Any] | None,
    fields_flat: Mapping[str, Any] | None,
) -> str | None:
    candidates: list[Any] = []

    if isinstance(summary, Mapping):
        candidates.extend(
            [
                summary.get("holder_name"),
                summary.get("consumer_name"),
                summary.get("consumer"),
            ]
        )

        labels = summary.get("labels")
        if isinstance(labels, Mapping):
            candidates.append(labels.get("holder_name"))
            candidates.append(labels.get("consumer_name"))

        normalized = summary.get("normalized")
        if isinstance(normalized, Mapping):
            candidates.append(normalized.get("holder_name"))

        meta = summary.get("meta")
        if isinstance(meta, Mapping):
            candidates.append(meta.get("holder_name"))
            candidates.append(meta.get("heading_guess"))

    if isinstance(fields_flat, Mapping):
        for key in ("holder_name", "consumer_name"):
            value = fields_flat.get(key)
            if value:
                candidates.append(value)

    for candidate in candidates:
        text = _extract_text(candidate)
        if text:
            return text

    return None


def _extract_issue_tags(tags_path: Path) -> tuple[str | None, list[str]]:
    payload = _load_json_payload(tags_path)
    issues: list[str] = []
    seen: set[str] = set()
    if isinstance(payload, Sequence):
        for entry in payload:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("kind") != "issue":
                continue
            issue_type = entry.get("type")
            if not isinstance(issue_type, str):
                continue
            trimmed = issue_type.strip()
            if not trimmed:
                continue
            if trimmed in seen:
                continue
            issues.append(trimmed)
            seen.add(trimmed)

    primary = issues[0] if issues else None
    return primary, issues


def _summarize_balance(balance_payload: Mapping[str, Any] | None) -> str | None:
    if not isinstance(balance_payload, Mapping):
        return None

    consensus = balance_payload.get("consensus")
    if isinstance(consensus, str) and consensus.strip():
        return consensus.strip()

    per_bureau = balance_payload.get("per_bureau")
    if isinstance(per_bureau, Mapping):
        for value in per_bureau.values():
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _prepare_bureau_payload_from_flat(
    *,
    account_number_values: Mapping[str, str],
    balance_values: Mapping[str, str],
    date_opened_values: Mapping[str, str],
    closed_date_values: Mapping[str, str],
    date_reported_values: Mapping[str, str],
    reported_bureaus: Iterable[str],
) -> dict[str, Any]:
    badges = [
        dict(_BUREAU_BADGES[bureau])
        for bureau in reported_bureaus
        if bureau in _BUREAU_BADGES
    ]

    if not badges:
        badges = [dict(_BUREAU_BADGES[bureau]) for bureau in _BUREAU_ORDER]

    displays = [value for value in account_number_values.values() if isinstance(value, str)]
    last4_info = _extract_last4(displays)

    balance_consensus = _resolve_majority_consensus(balance_values)

    return {
        "last4": last4_info,
        "balance_owed": {
            "per_bureau": dict(balance_values),
            **({"consensus": balance_consensus} if balance_consensus else {}),
        },
        "dates": {
            "date_opened": dict(date_opened_values),
            "closed_date": dict(closed_date_values),
            "date_reported": dict(date_reported_values),
        },
        "bureau_badges": badges,
    }


def _stringify_flat_value(value: Any) -> str | None:
    if isinstance(value, (int, float, Decimal)):
        return str(value)

    if isinstance(value, Mapping):
        for key in (
            "display",
            "text",
            "label",
            "normalized",
            "value",
            "amount",
            "raw",
            "name",
        ):
            candidate = value.get(key)
            text = _stringify_flat_value(candidate)
            if text:
                return text
        return None

    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None

    return None


def _flat_lookup(
    flat_payload: Mapping[str, Any] | None, field: str, bureau: str | None = None
) -> Any:
    if not isinstance(flat_payload, Mapping):
        return None

    if bureau is not None:
        per_bureau = flat_payload.get("per_bureau")
        if isinstance(per_bureau, Mapping):
            candidate = per_bureau.get(bureau)
            if isinstance(candidate, Mapping):
                inner = candidate.get(field)
                if inner is not None:
                    return inner
            elif candidate is not None:
                return candidate

        bureau_payload = flat_payload.get(bureau)
        if isinstance(bureau_payload, Mapping):
            direct = bureau_payload.get(field)
            if direct is not None:
                return direct
            nested = bureau_payload.get("fields")
            if isinstance(nested, Mapping):
                candidate = nested.get(field)
                if candidate is not None:
                    return candidate

        for key in (
            f"{bureau}_{field}",
            f"{field}_{bureau}",
            f"{bureau}.{field}",
            f"{field}.{bureau}",
        ):
            if key in flat_payload:
                return flat_payload[key]

    field_payload = flat_payload.get(field)
    if isinstance(field_payload, Mapping):
        if bureau is not None:
            candidate = field_payload.get(bureau)
            if candidate is not None:
                return candidate
        per_bureau = field_payload.get("per_bureau")
        if isinstance(per_bureau, Mapping) and bureau is not None:
            candidate = per_bureau.get(bureau)
            if candidate is not None:
                return candidate
        value = field_payload.get("value")
        if value is not None and bureau is None:
            return value

    return None


def _collect_flat_field_per_bureau(
    flat_payload: Mapping[str, Any] | None, field: str
) -> dict[str, str]:
    values: dict[str, str] = {}
    for bureau in _BUREAU_ORDER:
        raw_value = _flat_lookup(flat_payload, field, bureau)
        text = _stringify_flat_value(raw_value)
        if text:
            values[bureau] = text
    return values


def _collect_flat_consensus(flat_payload: Mapping[str, Any] | None, field: str) -> str | None:
    values: set[str] = set()
    if not isinstance(flat_payload, Mapping):
        return None

    value = _stringify_flat_value(flat_payload.get(field))
    if value:
        values.add(value)

    for bureau in _BUREAU_ORDER:
        bureau_value = _stringify_flat_value(_flat_lookup(flat_payload, field, bureau))
        if bureau_value:
            values.add(bureau_value)

    if len(values) == 1:
        return next(iter(values))
    return None


def _determine_reported_bureaus(
    summary: Mapping[str, Any] | None,
    flat_payload: Mapping[str, Any] | None,
) -> list[str]:
    reported: list[str] = []

    if isinstance(summary, Mapping):
        bureaus = summary.get("bureaus")
        if isinstance(bureaus, Sequence):
            for entry in bureaus:
                if isinstance(entry, str):
                    normalized = entry.strip().lower()
                    if normalized in _BUREAU_BADGES and normalized not in reported:
                        reported.append(normalized)

    if isinstance(flat_payload, Mapping):
        for bureau in _BUREAU_ORDER:
            if bureau in reported:
                continue
            candidate = _flat_lookup(flat_payload, "account_number_display", bureau)
            if _stringify_flat_value(candidate):
                reported.append(bureau)

    return reported


def _normalize_per_bureau(source: Mapping[str, Any] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for bureau in _BUREAU_ORDER:
        raw_value: Any | None = None
        if isinstance(source, Mapping):
            raw_value = source.get(bureau)
        if isinstance(raw_value, str):
            value = raw_value.strip() or None
        elif raw_value is not None:
            value = str(raw_value).strip() or None
        else:
            value = None
        normalized[bureau] = value if value else "--"
    return normalized


def _normalize_consensus_text(value: Any) -> str:
    if isinstance(value, str):
        trimmed = value.strip()
    elif value is None:
        trimmed = ""
    else:
        trimmed = str(value).strip()
    return trimmed if trimmed else "--"


def build_display_payload(
    *,
    holder_name: str,
    primary_issue: str,
    account_number_per_bureau: Mapping[str, str],
    account_number_consensus: str | None,
    account_type_per_bureau: Mapping[str, str],
    account_type_consensus: str | None,
    status_per_bureau: Mapping[str, str],
    status_consensus: str | None,
    balance_per_bureau: Mapping[str, str],
    date_opened_per_bureau: Mapping[str, str],
    closed_date_per_bureau: Mapping[str, str],
) -> dict[str, Any]:
    account_number_consensus_text = _normalize_consensus_text(account_number_consensus)
    account_type_consensus_text = _normalize_consensus_text(account_type_consensus)
    status_consensus_text = _normalize_consensus_text(status_consensus)

    return {
        "display_version": _DISPLAY_SCHEMA_VERSION,
        "holder_name": holder_name,
        "primary_issue": primary_issue,
        "account_number": {
            "per_bureau": dict(account_number_per_bureau),
            "consensus": account_number_consensus_text,
        },
        "account_type": {
            "per_bureau": dict(account_type_per_bureau),
            "consensus": account_type_consensus_text,
        },
        "status": {
            "per_bureau": dict(status_per_bureau),
            "consensus": status_consensus_text,
        },
        "balance_owed": {"per_bureau": dict(balance_per_bureau)},
        "date_opened": dict(date_opened_per_bureau),
        "closed_date": dict(closed_date_per_bureau),
    }


def _build_compact_display(
    *,
    holder_name: str | None,
    primary_issue: str | None,
    display_payload: Mapping[str, Any],
) -> dict[str, Any]:
    def _copy_account_section(
        source: Mapping[str, Any] | None, *, include_consensus: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"per_bureau": {}}
        if isinstance(source, Mapping):
            per_bureau = source.get("per_bureau")
            if isinstance(per_bureau, Mapping):
                payload["per_bureau"] = dict(per_bureau)
            if include_consensus:
                consensus = source.get("consensus")
                if consensus is not None:
                    payload["consensus"] = consensus if isinstance(consensus, str) else str(consensus)
        return payload

    def _bureau_dates(source: Mapping[str, Any] | None) -> dict[str, Any]:
        return dict(source) if isinstance(source, Mapping) else {}

    return {
        "display_version": display_payload.get(
            "display_version", _DISPLAY_SCHEMA_VERSION
        ),
        "holder_name": holder_name,
        "primary_issue": primary_issue,
        "account_number": _copy_account_section(
            display_payload.get("account_number"), include_consensus=True
        ),
        "account_type": _copy_account_section(
            display_payload.get("account_type"), include_consensus=True
        ),
        "status": _copy_account_section(
            display_payload.get("status"), include_consensus=True
        ),
        "balance_owed": _copy_account_section(
            display_payload.get("balance_owed"), include_consensus=False
        ),
        "date_opened": _bureau_dates(display_payload.get("date_opened")),
        "closed_date": _bureau_dates(display_payload.get("closed_date")),
    }


def build_pack_doc(
    *,
    sid: str,
    account_id: str,
    creditor_name: str | None,
    account_type: str | None,
    status: str | None,
    bureau_summary: Mapping[str, Any],
    holder_name: str | None,
    primary_issue: str | None,
    display_payload: Mapping[str, Any],
    pointers: Mapping[str, str],
    issues: Sequence[str] | None,
) -> dict[str, Any]:
    payload = {
        "sid": sid,
        "account_id": account_id,
        "creditor_name": creditor_name,
        "account_type": account_type,
        "status": status,
        "last4": bureau_summary["last4"],
        "balance_owed": bureau_summary["balance_owed"],
        "dates": bureau_summary["dates"],
        "bureau_badges": bureau_summary["bureau_badges"],
        "holder_name": holder_name,
        "primary_issue": primary_issue,
        "display": dict(display_payload),
        "pointers": dict(pointers),
        "questions": list(_QUESTION_SET),
        "claim_field_links": _claim_field_links_payload(),
    }
    if issues:
        payload["issues"] = list(issues)
    return payload


def build_lean_pack_doc(
    *,
    holder_name: str | None,
    primary_issue: str | None,
    display_payload: Mapping[str, Any],
    pointers: Mapping[str, str],
    questions: Sequence[Any],
) -> dict[str, Any]:
    display = _build_compact_display(
        holder_name=holder_name,
        primary_issue=primary_issue,
        display_payload=display_payload,
    )

    questions_payload = _coerce_question_list(questions)

    return {
        "holder_name": holder_name,
        "primary_issue": primary_issue,
        "display": display,
        "questions": questions_payload,
        "pointers": dict(pointers),
        "claim_field_links": _claim_field_links_payload(),
    }


def build_stage_pack_doc(
    *,
    account_id: str,
    holder_name: str | None,
    primary_issue: str | None,
    display_payload: Mapping[str, Any],
) -> dict[str, Any]:
    display = _build_compact_display(
        holder_name=holder_name,
        primary_issue=primary_issue,
        display_payload=display_payload,
    )

    return {
        "account_id": account_id,
        "holder_name": holder_name,
        "primary_issue": primary_issue,
        "display": display,
        "claim_field_links": _claim_field_links_payload(),
    }


def _has_meaningful_text(value: Any, *, treat_unknown: bool = False) -> bool:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return False
        if normalized == "--":
            return False
        if treat_unknown and normalized.lower() == "unknown":
            return False
        return True
    return value is not None


def _mapping_has_meaningful_values(
    mapping: Mapping[str, Any] | None, *, treat_unknown: bool = False
) -> bool:
    if not isinstance(mapping, Mapping):
        return False
    for value in mapping.values():
        if _has_meaningful_text(value, treat_unknown=treat_unknown):
            return True
    return False


def _has_meaningful_display(display: Mapping[str, Any] | None) -> bool:
    if not isinstance(display, Mapping):
        return False

    if _has_meaningful_text(display.get("holder_name"), treat_unknown=True):
        return True
    if _has_meaningful_text(display.get("primary_issue"), treat_unknown=True):
        return True

    account_number = display.get("account_number")
    if isinstance(account_number, Mapping):
        if _has_meaningful_text(account_number.get("consensus")):
            return True
        if _mapping_has_meaningful_values(account_number.get("per_bureau")):
            return True

    account_type = display.get("account_type")
    if isinstance(account_type, Mapping):
        if _has_meaningful_text(account_type.get("consensus"), treat_unknown=True):
            return True
        if _mapping_has_meaningful_values(account_type.get("per_bureau"), treat_unknown=True):
            return True

    status = display.get("status")
    if isinstance(status, Mapping):
        if _has_meaningful_text(status.get("consensus"), treat_unknown=True):
            return True
        if _mapping_has_meaningful_values(status.get("per_bureau"), treat_unknown=True):
            return True

    balance = display.get("balance_owed")
    if isinstance(balance, Mapping) and _mapping_has_meaningful_values(
        balance.get("per_bureau")
    ):
        return True

    if _mapping_has_meaningful_values(display.get("date_opened")):
        return True
    if _mapping_has_meaningful_values(display.get("closed_date")):
        return True

    return False


def _stage_payload_has_meaningful_data(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False

    if _has_meaningful_text(payload.get("holder_name"), treat_unknown=True):
        return True
    if _has_meaningful_text(payload.get("primary_issue"), treat_unknown=True):
        return True

    for key in ("creditor_name", "account_type", "status"):
        if _has_meaningful_text(payload.get(key), treat_unknown=True):
            return True

    display_payload = payload.get("display")
    if _has_meaningful_display(display_payload):
        return True

    return False


def _safe_account_dirname(account_id: str, fallback: str) -> str:
    account_id = account_id.strip()
    if not account_id:
        return fallback
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", account_id)
    return sanitized or fallback


def _build_stage_manifest(
    *,
    sid: str,
    stage_name: str,
    run_dir: Path,
    stage_packs_dir: Path,
    stage_responses_dir: Path,
    stage_index_path: Path,
    question_set: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    stage_dir = stage_index_path.parent
    pack_entries: list[dict[str, Any]] = []
    pack_index_entries: list[dict[str, Any]] = []

    if stage_packs_dir.is_dir():
        pack_paths = sorted(
            (
                path
                for path in stage_packs_dir.iterdir()
                if path.is_file() and path.suffix == ".json"
            ),
            key=_account_sort_key,
        )

        for pack_path in pack_paths:
            payload = _load_json_payload(pack_path)
            holder_name: str | None = None
            primary_issue: str | None = None
            has_questions = False

            if isinstance(payload, Mapping):
                holder_name = _optional_str(payload.get("holder_name"))
                primary_issue = _optional_str(payload.get("primary_issue"))
                questions = payload.get("questions")
                if isinstance(questions, Sequence) and not isinstance(
                    questions, (str, bytes, bytearray)
                ):
                    has_questions = len(questions) > 0

            if not has_questions:
                has_questions = bool(_QUESTION_SET)

            account_id = None
            if isinstance(payload, Mapping):
                account_id = _optional_str(payload.get("account_id"))

            if not account_id:
                account_id = pack_path.stem

            display_payload = None
            if isinstance(payload, Mapping):
                raw_display = payload.get("display")
                if isinstance(raw_display, Mapping):
                    display_payload = raw_display

            stage_relative_path = _relative_to_stage_dir(pack_path, stage_dir)
            run_relative_path = _relative_to_run_dir(pack_path, run_dir)

            pack_entry: dict[str, Any] = {
                "account_id": account_id,
                "holder_name": holder_name,
                "primary_issue": primary_issue,
                "path": run_relative_path,
                "bytes": os.path.getsize(pack_path),
                "has_questions": has_questions,
            }

            if display_payload is not None:
                pack_entry["display"] = display_payload

            pack_entry["pack_path"] = run_relative_path
            pack_entry["pack_path_rel"] = stage_relative_path
            pack_entry["file"] = run_relative_path

            sha1_digest = _safe_sha1(pack_path)
            if sha1_digest:
                pack_entry["sha1"] = sha1_digest

            pack_entries.append(pack_entry)
            pack_index_entries.append({"account": account_id, "file": stage_relative_path})

    responses_count = _count_frontend_responses(stage_responses_dir)
    responses_dir_value = _relative_to_run_dir(stage_responses_dir, run_dir)
    responses_dir_rel = _relative_to_stage_dir(stage_responses_dir, stage_dir)
    packs_dir_value = _relative_to_run_dir(stage_packs_dir, run_dir)
    packs_dir_rel = _relative_to_stage_dir(stage_packs_dir, stage_dir)
    index_path_value = _relative_to_run_dir(stage_index_path, run_dir)
    index_rel_value = _relative_to_stage_dir(stage_index_path, stage_dir)

    questions_payload = list(question_set) if question_set is not None else list(_QUESTION_SET)

    manifest_core: dict[str, Any] = {
        "sid": sid,
        "stage": stage_name,
        "schema_version": "1.0",
        "counts": {
            "packs": len(pack_entries),
            "responses": responses_count,
        },
        "packs": pack_entries,
        "responses_dir": responses_dir_value,
        "responses_dir_rel": responses_dir_rel,
        "packs_dir": packs_dir_value,
        "packs_dir_rel": packs_dir_rel,
        "index_path": index_path_value,
        "index_rel": index_rel_value,
        "packs_count": len(pack_entries),
        "questions": questions_payload,
        "packs_index": pack_index_entries,
    }

    generated_at = _now_iso()
    built_at = generated_at
    existing_manifest = _load_json_payload(stage_index_path)
    if isinstance(existing_manifest, Mapping):
        previous_core = dict(existing_manifest)
        previous_generated = previous_core.pop("generated_at", None)
        previous_built = previous_core.pop("built_at", None)
        if previous_core == manifest_core:
            if isinstance(previous_generated, str):
                generated_at = previous_generated
            if isinstance(previous_built, str):
                built_at = previous_built

    if not built_at:
        built_at = generated_at

    manifest_payload = {**manifest_core, "generated_at": generated_at, "built_at": built_at}
    _write_json_if_changed(stage_index_path, manifest_payload)
    log.info(
        "wrote review index sid=%s (count=%d) path=%s",
        sid,
        len(pack_entries),
        stage_index_path,
    )

    return manifest_payload


def _migrate_legacy_frontend_root_packs(
    *,
    sid: str,
    stage_name: str,
    run_dir: Path,
    stage_dir: Path,
    stage_packs_dir: Path,
    stage_responses_dir: Path,
    stage_index_path: Path,
    redirect_stub_path: Path,
) -> Mapping[str, Any] | None:
    canonical = get_frontend_review_paths(str(run_dir))
    legacy_glob = os.path.join(canonical["frontend_base"], "idx-*.json")

    moved = 0
    for legacy_path in glob.glob(legacy_glob):
        source = Path(legacy_path)
        if not source.is_file():
            continue

        destination = stage_packs_dir / source.name
        if destination.exists():
            log.warning(
                "FRONTEND_LEGACY_MIGRATE_EXISTS sid=%s source=%s target=%s",
                sid,
                source,
                destination,
            )
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.move(str(source), str(destination))
        except (OSError, shutil.Error):
            log.warning(
                "FRONTEND_LEGACY_MIGRATE_FAILED sid=%s source=%s target=%s",
                sid,
                source,
                destination,
                exc_info=True,
            )
            continue

        moved += 1

    if not moved:
        return None

    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_packs_dir.mkdir(parents=True, exist_ok=True)
    stage_responses_dir.mkdir(parents=True, exist_ok=True)
    stage_index_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_payload = _build_stage_manifest(
        sid=sid,
        stage_name=stage_name,
        run_dir=run_dir,
        stage_packs_dir=stage_packs_dir,
        stage_responses_dir=stage_responses_dir,
        stage_index_path=stage_index_path,
        question_set=_QUESTION_SET,
    )
    _ensure_frontend_index_redirect_stub(redirect_stub_path, force=True)

    log.info("FRONTEND_LEGACY_PACK_MIGRATION sid=%s moved=%d", sid, moved)
    return manifest_payload


def generate_frontend_packs_for_run(
    sid: str,
    *,
    runs_root: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build customer-facing account packs for ``sid``."""

    base_root = _resolve_runs_root(runs_root)
    run_dir = base_root / sid
    accounts_dir = run_dir / "cases" / "accounts"

    def _env_override(*names: str, default: str | None = None) -> str | None:
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        return default

    stage_name = _env_override("FRONTEND_STAGE_NAME", "FRONTEND_STAGE", default="review")

    lock_state, lock_path = _acquire_frontend_build_lock(run_dir, sid)
    if lock_state == "locked":
        canonical_paths = get_frontend_review_paths(str(run_dir))
        packs_dir_candidate = canonical_paths.get("packs_dir")
        packs_dir_path = (
            Path(packs_dir_candidate)
            if isinstance(packs_dir_candidate, str) and packs_dir_candidate
            else run_dir / "frontend" / "review" / "packs"
        )
        packs_dir_str = str(packs_dir_path.absolute())
        log.info("FRONTEND_BUILD_SKIP sid=%s reason=%s", sid, "locked")
        return {
            "status": "locked",
            "packs_count": 0,
            "empty_ok": True,
            "built": False,
            "packs_dir": packs_dir_str,
            "last_built_at": None,
            "skip_reason": "locked",
        }
    lock_acquired = lock_state == "acquired"

    try:
        config = load_frontend_stage_config(run_dir)

        stage_dir = config.stage_dir
        stage_packs_dir = config.packs_dir
        stage_responses_dir = config.responses_dir
        stage_index_path = config.index_path
        debug_packs_dir = stage_dir / "debug"
    
        canonical_paths = ensure_frontend_review_dirs(str(run_dir))
    
        _log_stage_paths(sid, config, canonical_paths)
    
        legacy_accounts_dir = run_dir / "frontend" / "accounts"
        if legacy_accounts_dir.is_dir():
            log.warning(
                "FRONTEND_LEGACY_ACCOUNTS_DIR sid=%s path=%s",
                sid,
                legacy_accounts_dir,
            )
    
        legacy_index_env = _env_override("FRONTEND_INDEX_PATH", "FRONTEND_INDEX")
        if legacy_index_env:
            candidate = Path(legacy_index_env)
            if not candidate.is_absolute():
                redirect_stub_path = run_dir / candidate
            else:
                redirect_stub_path = candidate
        else:
            redirect_stub_path = Path(
                canonical_paths.get("legacy_index", canonical_paths["index"])
            )
    
        _migrate_legacy_frontend_root_packs(
            sid=sid,
            stage_name=stage_name,
            run_dir=run_dir,
            stage_dir=stage_dir,
            stage_packs_dir=stage_packs_dir,
            stage_responses_dir=stage_responses_dir,
            stage_index_path=stage_index_path,
            redirect_stub_path=redirect_stub_path,
        )
        packs_dir_str = str(stage_packs_dir.absolute())
    
        frontend_autorun_enabled = _env_flag_enabled("FRONTEND_STAGE_AUTORUN", True)
        review_autorun_enabled = _env_flag_enabled("REVIEW_STAGE_AUTORUN", True)
        if not (frontend_autorun_enabled and review_autorun_enabled):
            if not frontend_autorun_enabled and not review_autorun_enabled:
                reason = "autorun_disabled"
            elif not frontend_autorun_enabled:
                reason = "frontend_stage_autorun_disabled"
            else:
                reason = "review_stage_autorun_disabled"
            log.info(
                "FRONTEND_AUTORUN_DISABLED sid=%s reason=%s",
                sid,
                reason,
            )
            return {
                "status": reason,
                "packs_count": 0,
                "empty_ok": True,
                "built": False,
                "packs_dir": packs_dir_str,
                "last_built_at": None,
                "autorun_disabled": True,
            }
    
        runflow_stage_start("frontend", sid=sid)
        current_account_id: str | None = None
        try:
            account_dirs: list[Path] = (
                sorted(
                    [path for path in accounts_dir.iterdir() if path.is_dir()],
                    key=_account_sort_key,
                )
                if accounts_dir.is_dir()
                else []
            )
            total_accounts = len(account_dirs)
    
            runflow_step(
                sid,
                "frontend",
                "frontend_review_start",
                metrics={"accounts": total_accounts},
            )
    
            if not _frontend_packs_enabled():
                fallback_manifest: Mapping[str, Any] | None = None
                if _frontend_review_create_empty_index_enabled():
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    os.makedirs(stage_packs_dir, exist_ok=True)
                    os.makedirs(stage_responses_dir, exist_ok=True)
                    stage_index_path.parent.mkdir(parents=True, exist_ok=True)
                    fallback_manifest = _build_stage_manifest(
                        sid=sid,
                        stage_name=stage_name,
                        run_dir=run_dir,
                        stage_packs_dir=stage_packs_dir,
                        stage_responses_dir=stage_responses_dir,
                        stage_index_path=stage_index_path,
                        question_set=_QUESTION_SET,
                    )
                    _ensure_frontend_index_redirect_stub(redirect_stub_path)
                    log.info("FRONTEND_EMPTY_INDEX_FALLBACK sid=%s", sid)

                responses_count = _emit_responses_scan(sid, stage_responses_dir)
                summary: dict[str, Any] = {
                    "packs_count": 0,
                    "responses_received": responses_count,
                    "empty_ok": True,
                    "reason": "disabled",
                }
                if fallback_manifest is not None:
                    summary["fallback_index"] = True
                runflow_step(
                    sid,
                    "frontend",
                    "frontend_review_no_candidates",
                    out={"reason": "disabled"},
                )
                record_frontend_responses_progress(
                    sid,
                    accounts_published=0,
                    answers_received=responses_count,
                    answers_required=0,
                )
                runflow_step(
                    sid,
                    "frontend",
                    "frontend_review_finish",
                    status="skipped",
                    metrics={"packs": 0},
                    out={"reason": "disabled"},
                )
                runflow_stage_end(
                    "frontend",
                    sid=sid,
                    status="skipped",
                    summary=summary,
                    empty_ok=True,
                )
                if isinstance(fallback_manifest, Mapping):
                    generated_at = fallback_manifest.get("generated_at")
                    fallback_last_built = generated_at if isinstance(generated_at, str) else None
                else:
                    fallback_last_built = None
                _log_done(
                    sid,
                    0,
                    status="skipped",
                    reason="disabled",
                    fallback_index=bool(fallback_manifest),
                )
                result = {
                    "status": "skipped",
                    "packs_count": 0,
                    "empty_ok": True,
                    "built": False,
                    "packs_dir": packs_dir_str,
                    "last_built_at": fallback_last_built,
                }
                if fallback_manifest is not None:
                    result["fallback_index"] = True
                _log_build_summary(
                    sid,
                    packs_count=0,
                    last_built_at=fallback_last_built,
                )
                return result
    
            stage_dir.mkdir(parents=True, exist_ok=True)
            os.makedirs(stage_packs_dir, exist_ok=True)
            os.makedirs(stage_responses_dir, exist_ok=True)
            stage_index_path.parent.mkdir(parents=True, exist_ok=True)
    
            lean_enabled = _frontend_packs_lean_enabled()
            debug_mirror_enabled = _frontend_packs_debug_mirror_enabled()
            stage_payload_mode = _resolve_stage_payload_mode()
            stage_payload_full = stage_payload_mode == _STAGE_PAYLOAD_MODE_FULL
    
            if not account_dirs:
                manifest_payload = _build_stage_manifest(
                    sid=sid,
                    stage_name=stage_name,
                    run_dir=run_dir,
                    stage_packs_dir=stage_packs_dir,
                    stage_responses_dir=stage_responses_dir,
                    stage_index_path=stage_index_path,
                    question_set=_QUESTION_SET,
                )
                _ensure_frontend_index_redirect_stub(redirect_stub_path)
                runflow_step(
                    sid,
                    "frontend",
                    "frontend_review_no_candidates",
                    metrics={"accounts": total_accounts},
                )
                runflow_step(
                    sid,
                    "frontend",
                    "frontend_review_finish",
                    metrics={"packs": 0},
                )
                responses_count = _emit_responses_scan(sid, stage_responses_dir)
                summary = {
                    "packs_count": 0,
                    "responses_received": responses_count,
                    "empty_ok": True,
                }
                record_frontend_responses_progress(
                    sid,
                    accounts_published=0,
                    answers_received=responses_count,
                    answers_required=0,
                )
                runflow_stage_end(
                    "frontend",
                    sid=sid,
                    summary=summary,
                    empty_ok=True,
                )
                _log_done(sid, 0, status="success")
                result = {
                    "status": "success",
                    "packs_count": 0,
                    "empty_ok": True,
                    "built": True,
                    "packs_dir": packs_dir_str,
                    "last_built_at": manifest_payload.get("generated_at"),
                }
                _log_build_summary(
                    sid,
                    packs_count=0,
                    last_built_at=manifest_payload.get("generated_at"),
                )
                return result
    
            if not force and stage_index_path.exists():
                existing = _load_json(stage_index_path)
                if existing:
                    pointer_backfill_required = _index_requires_pointer_backfill(
                        existing, run_dir
                    )
                    if pointer_backfill_required:
                        log.info("FRONTEND_PACK_POINTER_BACKFILL sid=%s", sid)
                    else:
                        packs_count = int(existing.get("packs_count", 0) or 0)
                        if not packs_count:
                            accounts = existing.get("accounts")
                            if isinstance(accounts, list):
                                packs_count = len(accounts)
                        log.debug(
                            "FRONTEND_PACKS_EXISTS sid=%s path=%s",
                            sid,
                            stage_index_path,
                        )
                        generated_at = existing.get("generated_at")
                        last_built = (
                            str(generated_at) if isinstance(generated_at, str) else None
                        )
                        _ensure_frontend_index_redirect_stub(redirect_stub_path)
                        if not debug_mirror_enabled and debug_packs_dir.is_dir():
                            for mirror_path in debug_packs_dir.glob("*.full.json"):
                                try:
                                    mirror_path.unlink()
                                except FileNotFoundError:
                                    continue
                                except OSError:  # pragma: no cover - defensive logging
                                    log.warning(
                                        "FRONTEND_PACK_DEBUG_MIRROR_UNLINK_FAILED path=%s",
                                        mirror_path,
                                        exc_info=True,
                                    )
                        if packs_count == 0:
                            runflow_step(
                                sid,
                                "frontend",
                                "frontend_review_no_candidates",
                                out={"reason": "cache"},
                            )
                        runflow_step(
                            sid,
                            "frontend",
                            "frontend_review_finish",
                            status="success",
                            metrics={"packs": packs_count},
                            out={"cache_hit": True},
                        )
                        responses_count = _emit_responses_scan(sid, stage_responses_dir)
                        summary = {
                            "packs_count": packs_count,
                            "responses_received": responses_count,
                            "empty_ok": packs_count == 0,
                            "cache_hit": True,
                        }
                        record_frontend_responses_progress(
                            sid,
                            accounts_published=packs_count,
                            answers_received=responses_count,
                            answers_required=packs_count,
                        )
                        runflow_stage_end(
                            "frontend",
                            sid=sid,
                            summary=summary,
                            empty_ok=packs_count == 0,
                        )
                        _log_done(sid, packs_count, status="success", cache_hit=True)
                        result = {
                            "status": "success",
                            "packs_count": packs_count,
                            "empty_ok": packs_count == 0,
                            "built": True,
                            "packs_dir": packs_dir_str,
                            "last_built_at": last_built,
                        }
                        _log_build_summary(
                            sid,
                            packs_count=packs_count,
                            last_built_at=last_built,
                        )
                        return result
    
            built_docs = 0
            unchanged_docs = 0
            skipped_missing = 0
            skip_reasons = {"missing_summary": 0}
            write_errors: list[tuple[str, Exception]] = []
            pack_count = 0
    
            for account_dir in account_dirs:
                summary_path = account_dir / "summary.json"
                summary = _load_json(summary_path)
                if not summary:
                    skipped_missing += 1
                    skip_reasons["missing_summary"] = skip_reasons.get("missing_summary", 0) + 1
                    log.warning(
                        "FRONTEND_PACK_MISSING_SUMMARY sid=%s path=%s",
                        sid,
                        summary_path,
                    )
                    continue
    
                account_id = str(summary.get("account_id") or account_dir.name)
                current_account_id = account_id
    
                flat_path = account_dir / "fields_flat.json"
                fields_flat_payload = _load_json(flat_path)
                if fields_flat_payload is None:
                    log.warning(
                        "FRONTEND_PACK_MISSING_FLAT sid=%s account=%s path=%s",
                        sid,
                        account_id,
                        flat_path,
                    )
    
                tags_path = account_dir / "tags.json"
                if not tags_path.exists():
                    log.warning(
                        "FRONTEND_PACK_MISSING_TAGS sid=%s account=%s path=%s",
                        sid,
                        account_id,
                        tags_path,
                    )
    
                labels = _extract_summary_labels(summary)
                holder_name = _derive_holder_name_from_summary(summary, fields_flat_payload)
                primary_issue, issues = _extract_issue_tags(tags_path)
                if not primary_issue:
                    primary_issue = "unknown"
    
                display_holder_name = _coerce_display_text(
                    holder_name or labels.get("creditor_name") or ""
                )
                if not display_holder_name:
                    display_holder_name = "Unknown"
                display_primary_issue = _coerce_display_text(primary_issue or "unknown")
    
                account_number_values_raw = _collect_flat_field_per_bureau(
                    fields_flat_payload, "account_number_display"
                )
                account_number_per_bureau = _normalize_per_bureau(account_number_values_raw)
                account_number_consensus = _resolve_account_number_consensus(
                    account_number_per_bureau
                )
    
                account_type_values_raw = _collect_flat_field_per_bureau(
                    fields_flat_payload, "account_type"
                )
                account_type_per_bureau = _normalize_per_bureau(account_type_values_raw)
                account_type_consensus = _resolve_majority_consensus(account_type_per_bureau)
                if account_type_consensus == "--":
                    fallback_account_type = labels.get("account_type") or _collect_flat_consensus(
                        fields_flat_payload, "account_type"
                    )
                    if fallback_account_type:
                        account_type_consensus = fallback_account_type
    
                status_values_raw = _collect_flat_field_per_bureau(
                    fields_flat_payload, "account_status"
                )
                status_per_bureau = _normalize_per_bureau(status_values_raw)
                status_consensus = _resolve_majority_consensus(status_per_bureau)
                if status_consensus == "--":
                    fallback_status = labels.get("status") or _collect_flat_consensus(
                        fields_flat_payload, "account_status"
                    )
                    if fallback_status:
                        status_consensus = fallback_status
    
                balance_values_raw = _collect_flat_field_per_bureau(
                    fields_flat_payload, "balance_owed"
                )
                balance_per_bureau = _normalize_per_bureau(balance_values_raw)
    
                date_opened_values_raw = _collect_flat_field_per_bureau(
                    fields_flat_payload, "date_opened"
                )
                date_opened_per_bureau = _normalize_per_bureau(date_opened_values_raw)
    
                closed_date_values_raw = _collect_flat_field_per_bureau(
                    fields_flat_payload, "closed_date"
                )
                closed_date_per_bureau = _normalize_per_bureau(closed_date_values_raw)
    
                date_reported_values_raw = _collect_flat_field_per_bureau(
                    fields_flat_payload, "date_reported"
                )
                date_reported_per_bureau = _normalize_per_bureau(date_reported_values_raw)
    
                reported_bureaus = _determine_reported_bureaus(summary, fields_flat_payload)
                bureau_summary = _prepare_bureau_payload_from_flat(
                    account_number_values=account_number_per_bureau,
                    balance_values=balance_per_bureau,
                    date_opened_values=date_opened_per_bureau,
                    closed_date_values=closed_date_per_bureau,
                    date_reported_values=date_reported_per_bureau,
                    reported_bureaus=reported_bureaus,
                )
    
                creditor_name_value = labels.get("creditor_name") or _stringify_flat_value(
                    _flat_lookup(fields_flat_payload, "creditor_name")
                ) or _extract_text(summary.get("creditor_name"))
                account_type_value = (
                    account_type_consensus
                    if account_type_consensus != "--"
                    else labels.get("account_type")
                )
                status_value = (
                    status_consensus if status_consensus != "--" else labels.get("status")
                )
    
                display_payload = build_display_payload(
                    holder_name=display_holder_name,
                    primary_issue=display_primary_issue,
                    account_number_per_bureau=account_number_per_bureau,
                    account_number_consensus=account_number_consensus,
                    account_type_per_bureau=account_type_per_bureau,
                    account_type_consensus=account_type_consensus,
                    status_per_bureau=status_per_bureau,
                    status_consensus=status_consensus,
                    balance_per_bureau=balance_per_bureau,
                    date_opened_per_bureau=date_opened_per_bureau,
                    closed_date_per_bureau=closed_date_per_bureau,
                )
    
                try:
                    relative_account_dir = account_dir.relative_to(run_dir).as_posix()
                except ValueError:
                    relative_account_dir = account_dir.as_posix()
    
                pointers = {
                    "summary": f"{relative_account_dir}/summary.json",
                    "tags": f"{relative_account_dir}/tags.json",
                    "flat": f"{relative_account_dir}/fields_flat.json",
                }
    
                full_pack_payload: dict[str, Any] | None = None
                need_full_payload = (
                    debug_mirror_enabled or not lean_enabled or stage_payload_full
                )
                if need_full_payload:
                    full_pack_payload = build_pack_doc(
                        sid=sid,
                        account_id=account_id,
                        creditor_name=creditor_name_value,
                        account_type=account_type_value,
                        status=status_value,
                        bureau_summary=bureau_summary,
                        holder_name=holder_name,
                        primary_issue=primary_issue,
                        display_payload=display_payload,
                        pointers=pointers,
                        issues=issues if issues else None,
                    )

                if stage_payload_full and full_pack_payload is not None:
                    stage_pack_payload = dict(full_pack_payload)
                else:
                    stage_pack_payload = build_stage_pack_doc(
                        account_id=account_id,
                        holder_name=display_holder_name,
                        primary_issue=display_primary_issue,
                        display_payload=display_payload,
                    )

                account_filename = _safe_account_dirname(account_id, account_dir.name)
                stage_pack_path = stage_packs_dir / f"{account_filename}.json"

                existing_stage_pack: Mapping[str, Any] | None = None
                if stage_pack_path.exists():
                    existing_payload = _load_json_payload(stage_pack_path)
                    if isinstance(existing_payload, Mapping):
                        existing_stage_pack = existing_payload

                stage_pack_payload["questions"] = _resolve_stage_pack_questions(
                    existing_pack=existing_stage_pack,
                    question_set=_QUESTION_SET,
                )

                if (
                    existing_stage_pack is not None
                    and not _stage_payload_has_meaningful_data(stage_pack_payload)
                ):
                    log.info(
                        "FRONTEND_STAGE_SKIP_PLACEHOLDER sid=%s account=%s",
                        sid,
                        account_id,
                    )
                    unchanged_docs += 1
                    pack_count += 1
                    continue

                log.info(
                    "writing pack sid=%s (acct=%s, type=%s, status=%s)",
                    sid,
                    account_id,
                    account_type_value or "unknown",
                    status_value or "unknown",
                )

                try:
                    stage_changed = _write_json_if_changed(
                        stage_pack_path, stage_pack_payload
                    )
                    changed = stage_changed
                    if debug_mirror_enabled and full_pack_payload is not None:
                        debug_packs_dir.mkdir(parents=True, exist_ok=True)
                        mirror_path = debug_packs_dir / f"{account_filename}.full.json"
                        _write_json_if_changed(mirror_path, full_pack_payload)
                    elif not debug_mirror_enabled:
                        mirror_path = debug_packs_dir / f"{account_filename}.full.json"
                        try:
                            mirror_path.unlink()
                        except FileNotFoundError:
                            pass
                        except OSError:  # pragma: no cover - defensive logging
                            log.warning(
                                "FRONTEND_PACK_DEBUG_MIRROR_UNLINK_FAILED path=%s",
                                mirror_path,
                                exc_info=True,
                            )
                except Exception as exc:
                    log.exception(
                        "FRONTEND_PACK_WRITE_FAILED sid=%s account=%s path=%s",
                        sid,
                        account_id,
                        stage_pack_path,
                    )
                    write_errors.append((account_id, exc))
                    continue
    
                if changed:
                    built_docs += 1
                else:
                    unchanged_docs += 1
    
                try:
                    relative_pack = stage_pack_path.relative_to(run_dir).as_posix()
                except ValueError:
                    relative_pack = str(stage_pack_path)
    
                pack_count += 1
    
                try:
                    relative_stage_pack = stage_pack_path.relative_to(run_dir).as_posix()
                except ValueError:
                    relative_stage_pack = str(stage_pack_path)
    
                if stage_changed and runflow_account_steps_enabled():
                    runflow_step(
                        sid,
                        "frontend",
                        "frontend_review_pack_created",
                        out={
                            "account_id": account_id,
                            "bytes": stage_pack_path.stat().st_size,
                            "path": relative_stage_pack,
                        },
                    )
    
            build_metrics = {
                "accounts": total_accounts,
                "built": built_docs,
                "skipped_missing": skipped_missing,
                "unchanged": unchanged_docs,
            }
            skip_summary = {key: value for key, value in skip_reasons.items() if value}
    
            generated_at = _now_iso()
            manifest_payload = _build_stage_manifest(
                sid=sid,
                stage_name=stage_name,
                run_dir=run_dir,
                stage_packs_dir=stage_packs_dir,
                stage_responses_dir=stage_responses_dir,
                stage_index_path=stage_index_path,
                question_set=_QUESTION_SET,
            )
            _ensure_frontend_index_redirect_stub(redirect_stub_path)
            done_status = "error" if write_errors else "success"
            _log_done(sid, pack_count, status=done_status)
    
            finish_out: dict[str, Any] = {
                "skip_reasons": skip_summary or None,
                "write_failures": len(write_errors) if write_errors else None,
            }
            finish_out = {key: value for key, value in finish_out.items() if value is not None}
    
            if pack_count == 0:
                runflow_step(
                    sid,
                    "frontend",
                    "frontend_review_no_candidates",
                )
            runflow_step(
                sid,
                "frontend",
                "frontend_review_finish",
                status=done_status,
                metrics={**build_metrics, "packs": pack_count},
                out=finish_out or None,
                error=(
                    {
                        "type": "PackWriteError",
                        "message": f"{len(write_errors)} pack writes failed",
                    }
                    if write_errors
                    else None
                ),
            )
    
            responses_count = _emit_responses_scan(sid, stage_responses_dir)
            summary = {
                "packs_count": pack_count,
                "responses_received": responses_count,
                "empty_ok": pack_count == 0,
                "skipped_missing": skipped_missing,
            }
            if built_docs:
                summary["built"] = built_docs
            if unchanged_docs:
                summary["unchanged"] = unchanged_docs
            if write_errors:
                summary["write_failures"] = len(write_errors)
            record_frontend_responses_progress(
                sid,
                accounts_published=pack_count,
                answers_received=responses_count,
                answers_required=pack_count,
            )
            runflow_stage_end(
                "frontend",
                sid=sid,
                summary=summary,
                empty_ok=pack_count == 0,
            )
    
            result = {
                "status": "success",
                "packs_count": pack_count,
                "empty_ok": pack_count == 0,
                "built": True,
                "packs_dir": packs_dir_str,
                "last_built_at": manifest_payload.get("generated_at"),
            }
            _log_build_summary(
                sid,
                packs_count=pack_count,
                last_built_at=manifest_payload.get("generated_at"),
            )
            return result
        except Exception as exc:
            runflow_step(
                sid,
                "frontend",
                "frontend_review_finish",
                status="error",
                out={
                    "account_id": current_account_id,
                    "error_class": exc.__class__.__name__,
                    "message": str(exc),
                },
                error={
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            runflow_stage_error(
                "frontend",
                sid=sid,
                error_type=exc.__class__.__name__,
                message=str(exc),
                traceback_tail=format_exception_tail(exc),
                hint=compose_hint("frontend pack generation", exc),
            )
            raise
    finally:
        if lock_acquired and lock_path is not None:
            _release_frontend_build_lock(lock_path, sid)
    
    
__all__ = ["generate_frontend_packs_for_run"]
