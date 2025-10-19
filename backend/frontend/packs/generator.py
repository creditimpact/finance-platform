"""Frontend pack generation helpers."""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.core.io.json_io import _atomic_write_json
from backend.core.paths.frontend_review import (
    ensure_frontend_review_dirs,
    get_frontend_review_paths,
)
from backend.core.runflow import runflow_step
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

log = logging.getLogger(__name__)

_BUREAU_BADGES: Mapping[str, Mapping[str, str]] = {
    "transunion": {"id": "transunion", "label": "TransUnion", "short_label": "TU"},
    "equifax": {"id": "equifax", "label": "Equifax", "short_label": "EF"},
    "experian": {"id": "experian", "label": "Experian", "short_label": "EX"},
}

_BUREAU_ORDER: tuple[str, ...] = ("transunion", "experian", "equifax")

_DISPLAY_SCHEMA_VERSION = "1.2"


_QUESTION_SET = [
    {"id": "ownership", "prompt": "Do you own this account?"},
    {"id": "recognize", "prompt": "Do you recognize this account on your report?"},
    {"id": "explanation", "prompt": "Anything else we should know about this account?"},
    {"id": "identity_theft", "prompt": "Is this account tied to identity theft?"},
]


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


def _derive_holder_name(meta: Mapping[str, Any] | None, raw_lines_path: Path) -> str | None:
    meta_heading = None
    if isinstance(meta, Mapping):
        meta_heading = _extract_text(meta.get("heading_guess"))
        if meta_heading:
            return meta_heading

    raw_payload = _load_json_payload(raw_lines_path)
    ordered_lines: list[str] = []
    if isinstance(raw_payload, Sequence) and not isinstance(raw_payload, (str, bytes, bytearray)):
        page_start = None
        line_start = None
        if isinstance(meta, Mapping):
            page_value = meta.get("page_start")
            if isinstance(page_value, int):
                page_start = page_value
            line_value = meta.get("line_start")
            if isinstance(line_value, int):
                line_start = line_value

        entries: list[tuple[int, str, int | None, int | None]] = []
        for index, entry in enumerate(raw_payload):
            text_value: str | None = None
            page_value: int | None = None
            line_value: int | None = None
            if isinstance(entry, Mapping):
                raw_text = entry.get("text")
                if isinstance(raw_text, str):
                    text_value = raw_text.strip()
                raw_page = entry.get("page")
                if isinstance(raw_page, int):
                    page_value = raw_page
                raw_line = entry.get("line")
                if isinstance(raw_line, int):
                    line_value = raw_line
            elif isinstance(entry, str):
                text_value = entry.strip()

            if not text_value:
                continue
            entries.append((index, text_value, page_value, line_value))

        if entries:
            def sort_key(item: tuple[int, str, int | None, int | None]) -> tuple[int, int]:
                index, _text, page_value, line_value = item
                if page_start is None and line_start is None:
                    return (0, index)

                penalty = 0
                if page_start is not None:
                    if page_value is None:
                        penalty += 500
                    else:
                        penalty += abs(page_value - page_start) * 100
                if line_start is not None:
                    if line_value is None:
                        penalty += 25
                    else:
                        penalty += abs(line_value - line_start)
                return (penalty, index)

            entries.sort(key=sort_key)
            ordered_lines = [item[1] for item in entries]

    if not ordered_lines:
        ordered_lines = [
            candidate.strip()
            for candidate in _load_raw_lines(raw_lines_path)
            if isinstance(candidate, str) and candidate.strip()
        ]

    return holder_name_from_raw_lines(ordered_lines)


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


def _prepare_bureau_payload(bureaus: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    badges = [dict(_BUREAU_BADGES[bureau]) for bureau in bureaus if bureau in _BUREAU_BADGES]

    displays = [
        str(payload.get("account_number_display") or "")
        for payload in bureaus.values()
        if isinstance(payload, Mapping)
    ]
    last4_info = _extract_last4(displays)

    balance_per_bureau, balance_consensus = _collect_field_per_bureau(bureaus, "balance_owed")

    date_opened_per_bureau, _ = _collect_field_per_bureau(bureaus, "date_opened")
    closed_date_per_bureau, _ = _collect_field_per_bureau(bureaus, "closed_date")
    date_reported_per_bureau, _ = _collect_field_per_bureau(bureaus, "date_reported")

    return {
        "last4": last4_info,
        "balance_owed": {
            "per_bureau": balance_per_bureau,
            **({"consensus": balance_consensus} if balance_consensus else {}),
        },
        "dates": {
            "date_opened": date_opened_per_bureau,
            "closed_date": closed_date_per_bureau,
            "date_reported": date_reported_per_bureau,
        },
        "bureau_badges": badges,
    }


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
    def _per_bureau_only(source: Mapping[str, Any] | None) -> dict[str, Any]:
        per_bureau: Mapping[str, Any] | None = None
        if isinstance(source, Mapping):
            per_bureau = source.get("per_bureau")
        if isinstance(per_bureau, Mapping):
            return {"per_bureau": dict(per_bureau)}
        return {"per_bureau": {}}

    def _bureau_dates(source: Mapping[str, Any] | None) -> dict[str, Any]:
        return dict(source) if isinstance(source, Mapping) else {}

    return {
        "display_version": display_payload.get(
            "display_version", _DISPLAY_SCHEMA_VERSION
        ),
        "holder_name": holder_name,
        "primary_issue": primary_issue,
        "account_number": _per_bureau_only(display_payload.get("account_number")),
        "account_type": _per_bureau_only(display_payload.get("account_type")),
        "status": _per_bureau_only(display_payload.get("status")),
        "balance_owed": _per_bureau_only(display_payload.get("balance_owed")),
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
    }


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
    pack_entries: list[dict[str, Any]] = []

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

            pack_entry: dict[str, Any] = {
                "account_id": account_id,
                "holder_name": holder_name,
                "primary_issue": primary_issue,
                "path": _relative_to_run_dir(pack_path, run_dir),
                "bytes": os.path.getsize(pack_path),
                "has_questions": has_questions,
            }

            if display_payload is not None:
                pack_entry["display"] = display_payload

            pack_entry["pack_path"] = pack_entry["path"]

            sha1_digest = _safe_sha1(pack_path)
            if sha1_digest:
                pack_entry["sha1"] = sha1_digest

            pack_entries.append(pack_entry)

    responses_count = _count_frontend_responses(stage_responses_dir)
    responses_dir_value = _relative_to_run_dir(stage_responses_dir, run_dir)

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
        "packs_count": len(pack_entries),
        "questions": questions_payload,
    }

    generated_at = _now_iso()
    existing_manifest = _load_json_payload(stage_index_path)
    if isinstance(existing_manifest, Mapping):
        previous_core = dict(existing_manifest)
        previous_generated = previous_core.pop("generated_at", None)
        if previous_core == manifest_core and isinstance(previous_generated, str):
            generated_at = previous_generated

    manifest_payload = {**manifest_core, "generated_at": generated_at}
    _write_json_if_changed(stage_index_path, manifest_payload)

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
            responses_count = _emit_responses_scan(sid, stage_responses_dir)
            summary = {
                "packs_count": 0,
                "responses_received": responses_count,
                "empty_ok": True,
                "reason": "disabled",
            }
            runflow_step(
                sid,
                "frontend",
                "frontend_review_no_candidates",
                out={"reason": "disabled"},
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
            _log_done(sid, 0, status="skipped", reason="disabled")
            result = {
                "status": "skipped",
                "packs_count": 0,
                "empty_ok": True,
                "built": False,
                "packs_dir": packs_dir_str,
                "last_built_at": None,
            }
            _log_build_summary(sid, packs_count=0, last_built_at=None)
            return result

        stage_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(stage_packs_dir, exist_ok=True)
        os.makedirs(stage_responses_dir, exist_ok=True)
        stage_index_path.parent.mkdir(parents=True, exist_ok=True)

        lean_enabled = _frontend_packs_lean_enabled()
        debug_mirror_enabled = _frontend_packs_debug_mirror_enabled()

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
        skip_reasons = {"missing_inputs": 0, "no_bureaus": 0}
        write_errors: list[tuple[str, Exception]] = []
        pack_count = 0

        for account_dir in account_dirs:
            summary = _load_json(account_dir / "summary.json")
            bureaus_payload = _load_json(account_dir / "bureaus.json")
            if not summary or not bureaus_payload:
                skipped_missing += 1
                skip_reasons["missing_inputs"] += 1
                continue

            bureaus: dict[str, Mapping[str, Any]] = {
                bureau: payload
                for bureau, payload in bureaus_payload.items()
                if isinstance(payload, Mapping)
            }
            if not bureaus:
                skipped_missing += 1
                skip_reasons["no_bureaus"] += 1
                continue

            account_id = str(summary.get("account_id") or account_dir.name)
            current_account_id = account_id
            labels = _extract_summary_labels(summary)
            bureau_summary = _prepare_bureau_payload(bureaus)
            first_bureau = next(iter(bureaus.values()), {})
            meta_payload = _load_json(account_dir / "meta.json")
            raw_path = account_dir / "raw_lines.json"
            tags_path = account_dir / "tags.json"
            holder_name = _derive_holder_name(meta_payload, raw_path)
            primary_issue, issues = _extract_issue_tags(tags_path)
            if not primary_issue:
                primary_issue = "unknown"

            display_holder_name = _coerce_display_text(holder_name)
            display_primary_issue = _coerce_display_text(primary_issue)
            
            creditor_name_value = labels.get("creditor_name") or _extract_text(
                first_bureau.get("creditor_name")
            )
            account_type_value = labels.get("account_type") or _extract_text(
                first_bureau.get("account_type")
            )
            status_value = labels.get("status") or _extract_text(
                first_bureau.get("account_status")
            )

            balance_payload = bureau_summary.get("balance_owed")
            per_bureau_balance_source = None
            if isinstance(balance_payload, Mapping):
                per_bureau_balance_source = balance_payload.get("per_bureau")
            balance_per_bureau = _normalize_per_bureau(per_bureau_balance_source)

            account_number_values = _collect_field_text_values(
                bureaus, "account_number_display"
            )
            account_number_per_bureau = _normalize_per_bureau(account_number_values)
            account_number_consensus = _resolve_account_number_consensus(
                account_number_per_bureau
            )

            account_type_values = _collect_field_text_values(bureaus, "account_type")
            account_type_per_bureau = _normalize_per_bureau(account_type_values)
            account_type_consensus = _resolve_majority_consensus(account_type_per_bureau)

            status_values = _collect_field_text_values(bureaus, "account_status")
            status_per_bureau = _normalize_per_bureau(status_values)
            status_consensus = _resolve_majority_consensus(status_per_bureau)

            dates_payload = bureau_summary.get("dates")
            date_opened_source = None
            closed_date_source = None
            if isinstance(dates_payload, Mapping):
                date_opened_source = dates_payload.get("date_opened")
                closed_date_source = dates_payload.get("closed_date")
            date_opened_per_bureau = _normalize_per_bureau(date_opened_source)
            closed_date_per_bureau = _normalize_per_bureau(closed_date_source)

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
                "meta": f"{relative_account_dir}/meta.json",
                "tags": f"{relative_account_dir}/tags.json",
                "raw": f"{relative_account_dir}/raw_lines.json",
                "bureaus": f"{relative_account_dir}/bureaus.json",
                "flat": f"{relative_account_dir}/fields_flat.json",
                "summary": f"{relative_account_dir}/summary.json",
            }

            full_pack_payload: dict[str, Any] | None = None
            if debug_mirror_enabled or not lean_enabled:
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

            if stage_changed:
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


__all__ = ["generate_frontend_packs_for_run"]
