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
from backend.core.runflow import runflow_end_stage, runflow_start_stage, runflow_step

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


def _frontend_packs_enabled() -> bool:
    value = os.getenv("ENABLE_FRONTEND_PACKS", "1")
    return value not in {"0", "false", "False"}


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
    packs_dir_str = str(frontend_dir.absolute())

    runflow_start_stage(sid, "frontend")

    try:
        if not _frontend_packs_enabled():
            log.info("PACKGEN_FRONTEND_SKIP sid=%s", sid)
            runflow_step(
                sid,
                "frontend",
                "scan_accounts",
                metrics={"accounts": 0},
            )
            runflow_end_stage(
                sid,
                "frontend",
                status="skipped",
                summary={"packs_count": 0, "reason": "disabled"},
            )
            return {
                "status": "skipped",
                "packs_count": 0,
                "empty_ok": True,
                "built": False,
                "packs_dir": packs_dir_str,
                "last_built_at": None,
            }

        accounts_output_dir.mkdir(parents=True, exist_ok=True)
        responses_dir.mkdir(parents=True, exist_ok=True)

        account_dirs = (
            sorted(
                [path for path in accounts_dir.iterdir() if path.is_dir()],
                key=_account_sort_key,
            )
            if accounts_dir.is_dir()
            else []
        )
        runflow_step(
            sid,
            "frontend",
            "scan_accounts",
            metrics={"accounts": len(account_dirs)},
        )

        if not account_dirs:
            generated_at = _now_iso()
            payload = {
                "sid": sid,
                "generated_at": generated_at,
                "accounts": [],
                "packs_count": 0,
                "questions": _QUESTION_SET,
            }
            _atomic_write_json(index_path, payload)
            runflow_step(
                sid,
                "frontend",
                "build_pack_docs",
                metrics={"built": 0, "skipped_missing": 0},
            )
            try:
                index_out = index_path.relative_to(run_dir).as_posix()
            except ValueError:
                index_out = str(index_path)
            runflow_step(
                sid,
                "frontend",
                "write_index",
                metrics={"packs": 0},
                out={"path": index_out},
            )
            runflow_end_stage(sid, "frontend", summary={"packs_count": 0})
            return {
                "status": "success",
                "packs_count": 0,
                "empty_ok": True,
                "built": True,
                "packs_dir": packs_dir_str,
                "last_built_at": generated_at,
            }

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
                generated_at = existing.get("generated_at")
                last_built = str(generated_at) if isinstance(generated_at, str) else None
                runflow_step(
                    sid,
                    "frontend",
                    "build_pack_docs",
                    metrics={"built": packs_count, "skipped_missing": 0},
                    out={"cache_hit": True},
                )
                try:
                    index_out = index_path.relative_to(run_dir).as_posix()
                except ValueError:
                    index_out = str(index_path)
                runflow_step(
                    sid,
                    "frontend",
                    "write_index",
                    metrics={"packs": packs_count},
                    out={"path": index_out},
                )
                runflow_end_stage(
                    sid,
                    "frontend",
                    summary={"packs_count": packs_count, "cache_hit": True},
                )
                return {
                    "status": "success",
                    "packs_count": packs_count,
                    "empty_ok": packs_count == 0,
                    "built": True,
                    "packs_dir": packs_dir_str,
                    "last_built_at": last_built,
                }

        packs: list[dict[str, Any]] = []
        built_docs = 0
        skipped_missing = 0

        for account_dir in account_dirs:
            summary = _load_json(account_dir / "summary.json")
            bureaus_payload = _load_json(account_dir / "bureaus.json")
            account_label = account_dir.name

            if not summary or not bureaus_payload:
                skipped_missing += 1
                runflow_step(
                    sid,
                    "frontend",
                    "build_pack_docs",
                    status="skipped",
                    account=account_label,
                    metrics={
                        "built": built_docs,
                        "skipped_missing": skipped_missing,
                    },
                    out={"reason": "missing_inputs"},
                )
                continue

            bureaus: dict[str, Mapping[str, Any]] = {
                bureau: payload
                for bureau, payload in bureaus_payload.items()
                if isinstance(payload, Mapping)
            }
            if not bureaus:
                skipped_missing += 1
                runflow_step(
                    sid,
                    "frontend",
                    "build_pack_docs",
                    status="skipped",
                    account=account_label,
                    metrics={
                        "built": built_docs,
                        "skipped_missing": skipped_missing,
                    },
                    out={"reason": "no_bureaus"},
                )
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

            built_docs += 1
            runflow_step(
                sid,
                "frontend",
                "build_pack_docs",
                account=account_id,
                metrics={
                    "built": built_docs,
                    "skipped_missing": skipped_missing,
                },
            )

            account_dirname = _safe_account_dirname(account_id, account_dir.name)
            pack_dir = accounts_output_dir / account_dirname
            pack_dir.mkdir(parents=True, exist_ok=True)
            pack_path = pack_dir / "pack.json"
            _atomic_write_json(pack_path, pack_payload)

            try:
                relative_pack = pack_path.relative_to(run_dir).as_posix()
            except ValueError:
                relative_pack = str(pack_path)

            runflow_step(
                sid,
                "frontend",
                "write_pack",
                account=account_id,
                out={"path": relative_pack},
            )

            packs.append(
                {
                    "account_id": account_id,
                    "pack_path": relative_pack,
                    "creditor_name": pack_payload["creditor_name"],
                    "account_type": pack_payload["account_type"],
                    "status": pack_payload["status"],
                    "bureau_badges": pack_payload["bureau_badges"],
                }
            )

        generated_at = _now_iso()
        pack_count = built_docs
        index_payload = {
            "sid": sid,
            "generated_at": generated_at,
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

        try:
            index_out = index_path.relative_to(run_dir).as_posix()
        except ValueError:
            index_out = str(index_path)

        runflow_step(
            sid,
            "frontend",
            "write_index",
            metrics={"packs": pack_count},
            out={"path": index_out},
        )

        runflow_end_stage(sid, "frontend", summary={"packs_count": pack_count})

        return {
            "status": "success",
            "packs_count": pack_count,
            "empty_ok": pack_count == 0,
            "built": True,
            "packs_dir": packs_dir_str,
            "last_built_at": generated_at,
        }
    except Exception as exc:
        runflow_step(
            sid,
            "frontend",
            "build_pack_docs",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        runflow_end_stage(
            sid,
            "frontend",
            status="error",
            summary={"error": exc.__class__.__name__, "phase": "generate"},
        )
        raise


__all__ = ["generate_frontend_packs_for_run"]
