"""Frontend pack generation helpers."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.core.io.json_io import _atomic_write_json
from backend.core.runflow import runflow_step
from backend.core.runflow.io import (
    compose_hint,
    format_exception_tail,
    runflow_stage_end,
    runflow_stage_error,
    runflow_stage_start,
)

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


def _frontend_packs_topn() -> int:
    raw_value = os.getenv("RUNFLOW_STEPS_FRONTEND_PACKS_TOPN")
    if raw_value is None:
        return 0
    try:
        value = int(raw_value.strip())
    except (TypeError, ValueError):
        return 0
    return max(value, 0)


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


def _emit_responses_scan(sid: str, responses_dir: Path) -> int:
    count = _count_frontend_responses(responses_dir)
    runflow_step(
        sid,
        "frontend",
        "responses_scan",
        metrics={"received": count},
    )
    return count


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

    runflow_stage_start("frontend", sid=sid)
    packs_topn = _frontend_packs_topn()

    try:
        if not _frontend_packs_enabled():
            runflow_step(
                sid,
                "frontend",
                "scan_accounts",
                metrics={"accounts": 0},
            )
            runflow_step(
                sid,
                "frontend",
                "build_pack_docs",
                status="skipped",
                metrics={"built": 0, "skipped_missing": 0},
                out={"reason": "disabled"},
            )
            responses_count = _emit_responses_scan(sid, responses_dir)
            summary = {
                "packs_count": 0,
                "responses_received": responses_count,
                "empty_ok": True,
                "reason": "disabled",
            }
            runflow_stage_end(
                "frontend",
                sid=sid,
                status="skipped",
                summary=summary,
                empty_ok=True,
            )
            _log_done(sid, 0, status="skipped", reason="disabled")
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
            _write_json_if_changed(index_path, payload)
            runflow_step(
                sid,
                "frontend",
                "build_pack_docs",
                metrics={"built": 0, "skipped_missing": 0},
                out={"reason": "no_accounts"},
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
            responses_count = _emit_responses_scan(sid, responses_dir)
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
                log.debug(
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
                responses_count = _emit_responses_scan(sid, responses_dir)
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
        unchanged_docs = 0
        skipped_missing = 0
        skip_reasons = {"missing_inputs": 0, "no_bureaus": 0}
        sampled_write_packs = 0
        write_errors: list[tuple[str, Exception]] = []

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
            labels = _extract_summary_labels(summary)
            bureau_summary = _prepare_bureau_payload(bureaus)
            first_bureau = next(iter(bureaus.values()), {})
            meta_payload = _load_json(account_dir / "meta.json")
            raw_path = account_dir / "raw_lines.json"
            tags_path = account_dir / "tags.json"
            holder_name = _derive_holder_name(meta_payload, raw_path)
            primary_issue, issues = _extract_issue_tags(tags_path)

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
                "holder_name": holder_name,
                "primary_issue": primary_issue,
                "pointers": pointers,
                "questions": list(_QUESTION_SET),
            }
            if issues:
                pack_payload["issues"] = issues

            account_dirname = _safe_account_dirname(account_id, account_dir.name)
            pack_dir = accounts_output_dir / account_dirname
            pack_path = pack_dir / "pack.json"

            try:
                pack_dir.mkdir(parents=True, exist_ok=True)
                changed = _write_json_if_changed(pack_path, pack_payload)
            except Exception as exc:
                log.exception(
                    "FRONTEND_PACK_WRITE_FAILED sid=%s account=%s path=%s",
                    sid,
                    account_id,
                    pack_path,
                )
                write_errors.append((account_id, exc))
                continue

            if changed:
                built_docs += 1
            else:
                unchanged_docs += 1

            try:
                relative_pack = pack_path.relative_to(run_dir).as_posix()
            except ValueError:
                relative_pack = str(pack_path)

            if changed and sampled_write_packs < packs_topn:
                runflow_step(
                    sid,
                    "frontend",
                    "write_pack",
                    account=account_id,
                    out={"path": relative_pack},
                )
                sampled_write_packs += 1

            packs.append(
                {
                    "account_id": account_id,
                    "pack_path": relative_pack,
                    "creditor_name": pack_payload["creditor_name"],
                    "account_type": pack_payload["account_type"],
                    "status": pack_payload["status"],
                    "holder_name": holder_name,
                    "primary_issue": primary_issue,
                    "balance_owed": _summarize_balance(pack_payload.get("balance_owed")),
                    "bureau_badges": pack_payload["bureau_badges"],
                }
            )

        build_metrics = {
            "built": built_docs,
            "skipped_missing": skipped_missing,
            "unchanged": unchanged_docs,
        }
        build_out: dict[str, Any] = {}
        skip_summary = {key: value for key, value in skip_reasons.items() if value}
        if skip_summary:
            build_out["skip_reasons"] = skip_summary
        if write_errors:
            build_out["write_failures"] = len(write_errors)
            build_out["failed_accounts"] = [acct for acct, _ in write_errors[:5]]

        build_status = "error" if write_errors else "success"
        error_payload = None
        if write_errors:
            error_payload = {
                "type": "PackWriteError",
                "message": f"{len(write_errors)} pack writes failed",
            }

        runflow_step(
            sid,
            "frontend",
            "build_pack_docs",
            status=build_status,
            metrics=build_metrics,
            out=build_out or None,
            error=error_payload,
        )

        generated_at = _now_iso()
        pack_count = len(packs)
        index_payload_base = {
            "sid": sid,
            "accounts": packs,
            "packs_count": pack_count,
            "questions": list(_QUESTION_SET),
        }

        existing_index = _load_json_payload(index_path)
        if isinstance(existing_index, Mapping):
            existing_base = {
                "sid": existing_index.get("sid"),
                "accounts": existing_index.get("accounts"),
                "packs_count": existing_index.get("packs_count"),
                "questions": existing_index.get("questions"),
            }
            if existing_base == index_payload_base:
                prior_generated = existing_index.get("generated_at")
                if isinstance(prior_generated, str):
                    generated_at = prior_generated

        index_payload = {**index_payload_base, "generated_at": generated_at}

        _write_json_if_changed(index_path, index_payload)

        done_status = "error" if write_errors else "success"
        _log_done(sid, pack_count, status=done_status)

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

        responses_count = _emit_responses_scan(sid, responses_dir)
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
