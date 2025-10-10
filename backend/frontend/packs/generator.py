"""Frontend pack generation helpers."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from backend.core.io.json_io import _atomic_write_json

log = logging.getLogger(__name__)

_BUREAU_BADGES: Mapping[str, Mapping[str, str]] = {
    "transunion": {"id": "transunion", "label": "TransUnion", "short_label": "TU"},
    "equifax": {"id": "equifax", "label": "Equifax", "short_label": "EF"},
    "experian": {"id": "experian", "label": "Experian", "short_label": "EX"},
}

_QUESTION_SET = [
    {"id": "ownership", "prompt": "Do you own this account?"},
    {"id": "recognize", "prompt": "Do you recognize this account on your report?"},
    {"id": "explanation", "prompt": "Anything else we should know about this account?"},
    {"id": "identity_theft", "prompt": "Is this account tied to identity theft?"},
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _default_runs_root() -> Path:
    root_env = os.getenv("RUNS_ROOT")
    return Path(root_env) if root_env else Path("runs")


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        return _default_runs_root()
    return Path(runs_root)


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


def _safe_account_dirname(account_id: str, fallback: str) -> str:
    account_id = account_id.strip()
    if not account_id:
        return fallback
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", account_id)
    return sanitized or fallback


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
    frontend_dir = run_dir / "frontend"
    accounts_output_dir = frontend_dir / "accounts"
    responses_dir = frontend_dir / "responses"
    index_path = frontend_dir / "index.json"

    accounts_output_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    if not accounts_dir.is_dir():
        payload = {
            "sid": sid,
            "generated_at": _now_iso(),
            "accounts": [],
            "packs_count": 0,
            "questions": _QUESTION_SET,
        }
        _atomic_write_json(index_path, payload)
        return {"status": "success", "packs_count": 0, "empty_ok": True}

    if not force and index_path.exists():
        existing = _load_json(index_path)
        if existing:
            packs_count = int(existing.get("packs_count", 0) or 0)
            if not packs_count:
                accounts = existing.get("accounts")
                if isinstance(accounts, list):
                    packs_count = len(accounts)
            log.info(
                "FRONTEND_PACKS_EXISTS sid=%s path=%s", sid, index_path
            )
            return {
                "status": "success",
                "packs_count": packs_count,
                "empty_ok": packs_count == 0,
            }

    account_dirs = [path for path in accounts_dir.iterdir() if path.is_dir()]

    packs: list[dict[str, Any]] = []
    pack_count = 0

    for account_dir in sorted(account_dirs, key=_account_sort_key):
        summary = _load_json(account_dir / "summary.json")
        bureaus_payload = _load_json(account_dir / "bureaus.json")

        if not summary or not bureaus_payload:
            continue

        bureaus: dict[str, Mapping[str, Any]] = {
            bureau: payload
            for bureau, payload in bureaus_payload.items()
            if isinstance(payload, Mapping)
        }
        if not bureaus:
            continue

        account_id = str(summary.get("account_id") or account_dir.name)
        labels = _extract_summary_labels(summary)
        bureau_summary = _prepare_bureau_payload(bureaus)
        first_bureau = next(iter(bureaus.values()), {})

        pack_payload = {
            "sid": sid,
            "account_id": account_id,
            "creditor_name": labels.get("creditor_name")
            or _extract_text(first_bureau.get("creditor_name")),
            "account_type": labels.get("account_type")
            or _extract_text(first_bureau.get("account_type")),
            "status": labels.get("status")
            or _extract_text(first_bureau.get("account_status")),
            "last4": bureau_summary["last4"],
            "balance_owed": bureau_summary["balance_owed"],
            "dates": bureau_summary["dates"],
            "bureau_badges": bureau_summary["bureau_badges"],
            "questions": _QUESTION_SET,
        }

        account_dirname = _safe_account_dirname(account_id, account_dir.name)
        pack_dir = accounts_output_dir / account_dirname
        pack_dir.mkdir(parents=True, exist_ok=True)
        pack_path = pack_dir / "pack.json"
        _atomic_write_json(pack_path, pack_payload)

        packs.append(
            {
                "account_id": account_id,
                "pack_path": str(pack_path.relative_to(run_dir)),
                "creditor_name": pack_payload["creditor_name"],
                "account_type": pack_payload["account_type"],
                "status": pack_payload["status"],
                "bureau_badges": pack_payload["bureau_badges"],
            }
        )
        pack_count += 1

    index_payload = {
        "sid": sid,
        "generated_at": _now_iso(),
        "accounts": packs,
        "packs_count": pack_count,
        "questions": _QUESTION_SET,
    }

    _atomic_write_json(index_path, index_payload)

    log.info(
        "FRONTEND_PACKS_GENERATED sid=%s packs=%d path=%s",
        sid,
        pack_count,
        index_path,
    )

    return {
        "status": "success",
        "packs_count": pack_count,
        "empty_ok": pack_count == 0,
    }


__all__ = ["generate_frontend_packs_for_run"]
