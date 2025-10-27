"""Frontend review pack builder using manifest-sourced account artifacts."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from backend.core.paths.frontend_review import ensure_frontend_review_dirs


log = logging.getLogger(__name__)


_BUREAU_ORDER = ("experian", "equifax", "transunion")
_TEXT_SENTINELS = {"", "--", "—", "unknown", "Unknown"}


def get_first(data: Mapping[str, Any], *keys: str) -> Any | None:
    """Return the first non-empty value found in ``data`` for ``keys``."""

    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if _is_missing_text(value):
            continue
        return value
    return None


@dataclass(frozen=True)
class AccountSource:
    idx_key: str
    numeric_id: int | None
    payload: Mapping[str, Any]


def build_review_packs(sid: str, manifest: Any) -> dict[str, Any]:
    """Build review packs for ``sid`` using manifest-provided artifacts."""

    manifest_path = getattr(manifest, "path", None)
    if manifest_path is None:
        raise ValueError("manifest path is required to resolve run directory")

    manifest_data = getattr(manifest, "data", None)
    if manifest_data is None:
        raise ValueError("manifest data is required")

    run_dir = Path(manifest_path).resolve().parent
    review_paths = ensure_frontend_review_dirs(str(run_dir))
    packs_dir = Path(review_paths["packs_dir"])
    index_path = Path(review_paths["index"])

    accounts_section = _extract_accounts_section(manifest_data)
    if not accounts_section:
        log.info("FRONTEND: no accounts registered in manifest sid=%s", sid)
        packs_dir.mkdir(parents=True, exist_ok=True)
        index_payload = _build_index_payload([], run_dir)
        index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"status": "empty", "packs_count": 0, "packs_dir": str(packs_dir)}

    sources = _coalesce_account_sources(accounts_section)

    pack_entries: list[tuple[str, Path, Mapping[str, Any]]] = []

    for source in sources:
        meta_payload = _load_json_payload(source, "meta", run_dir)
        fields_payload = _load_json_payload(source, "flat", run_dir)
        tags_payload = _load_json_payload(source, "tags", run_dir)
        bureaus_payload = _load_json_payload(source, "bureaus", run_dir)

        pack_payload = _build_pack_payload(
            sid=sid,
            source=source,
            meta=meta_payload,
            fields_flat=fields_payload,
            tags=tags_payload,
            bureaus=bureaus_payload,
        )

        packs_dir.mkdir(parents=True, exist_ok=True)
        pack_path = packs_dir / f"{source.idx_key}.json"
        pack_path.write_text(json.dumps(pack_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        account_number = pack_payload.get("account", {}).get("account_number")
        account_type = pack_payload.get("account", {}).get("type")
        status = pack_payload.get("account", {}).get("status")
        log.info(
            "FRONTEND: wrote pack %s (acct=%s, type=%s, status=%s)",
            f"{source.idx_key}.json",
            account_number or "",
            account_type or "",
            status or "",
        )

        pack_entries.append((source.idx_key, pack_path, pack_payload))

    index_payload = _build_index_payload(pack_entries, run_dir)
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "success",
        "packs_count": len(pack_entries),
        "packs_dir": str(packs_dir),
        "index": str(index_path),
    }


def _extract_accounts_section(manifest_data: Mapping[str, Any]) -> Mapping[str, Any]:
    artifacts = manifest_data.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return {}
    cases = artifacts.get("cases")
    if not isinstance(cases, Mapping):
        return {}
    accounts = cases.get("accounts")
    return accounts if isinstance(accounts, Mapping) else {}


def _coalesce_account_sources(accounts_section: Mapping[str, Any]) -> list[AccountSource]:
    by_number: dict[int, MutableMapping[str, Any]] = {}
    labels: dict[int, str] = {}
    others: list[AccountSource] = []

    for raw_key, payload in accounts_section.items():
        if not isinstance(payload, Mapping):
            continue

        numeric_id: int | None = None
        idx_key: str | None = None

        if isinstance(raw_key, str) and raw_key.startswith("idx-"):
            match = re.match(r"idx-(\d+)", raw_key)
            if match:
                numeric_id = int(match.group(1))
                idx_key = raw_key
        if numeric_id is None:
            try:
                numeric_id = int(str(raw_key))
            except (TypeError, ValueError):
                numeric_id = None

        if numeric_id is None:
            others.append(AccountSource(idx_key=str(raw_key), numeric_id=None, payload=payload))
            continue

        entry = by_number.setdefault(numeric_id, {})
        entry.update(payload)
        if idx_key:
            labels[numeric_id] = idx_key
        elif numeric_id not in labels:
            labels[numeric_id] = f"idx-{numeric_id:03d}"

    merged: list[AccountSource] = []
    for numeric_id in sorted(by_number):
        idx_key = labels[numeric_id]
        merged.append(AccountSource(idx_key=idx_key, numeric_id=numeric_id, payload=dict(by_number[numeric_id])))

    merged.extend(others)
    return merged


def _resolve_entry_path(source: AccountSource, key: str, run_dir: Path) -> Path | None:
    raw_value = source.payload.get(key)
    if not raw_value:
        return None
    candidate = Path(str(raw_value))
    if candidate.is_absolute():
        return candidate

    dir_raw = source.payload.get("dir")
    if dir_raw:
        base = Path(str(dir_raw))
        if not base.is_absolute():
            base = (run_dir / base).resolve()
        return (base / candidate).resolve()

    return (run_dir / candidate).resolve()


def _load_json_payload(source: AccountSource, kind: str, run_dir: Path) -> Any:
    path = _resolve_entry_path(source, kind, run_dir)
    if path is None:
        log.debug("FRONTEND: missing %s for account %s", kind, source.idx_key)
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        log.debug("FRONTEND: missing %s for account %s at %s", kind, source.idx_key, path)
        return {}
    except Exception:  # pragma: no cover - defensive logging
        log.exception(
            "FRONTEND: failed to read %s for account %s from %s",
            kind,
            source.idx_key,
            path,
        )
        return {}

    log.debug("FRONTEND: loaded %s for account %s from %s", kind, source.idx_key, path)
    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, Sequence):
        return list(payload)
    return {}


def _build_pack_payload(
    *,
    sid: str,
    source: AccountSource,
    meta: Mapping[str, Any],
    fields_flat: Mapping[str, Any],
    tags: Mapping[str, Any] | Sequence[Any],
    bureaus: Mapping[str, Any],
) -> Mapping[str, Any]:
    account_info = _build_account_section(source, meta, fields_flat, tags)
    bureau_summary = _build_bureau_summary(bureaus)

    return {
        "sid": sid,
        "account": account_info,
        "bureau_summary": bureau_summary,
        "attachments_policy": {"gov_id_and_poa_default": True},
        "claims_menu": _default_claims_menu(),
    }


def _build_account_section(
    source: AccountSource,
    meta: Mapping[str, Any],
    fields_flat: Mapping[str, Any],
    tags: Mapping[str, Any] | Sequence[Any],
) -> Mapping[str, Any]:
    furnisher = _normalize_text(
        get_first(meta, "furnisher_name", "creditor_name", "name")
    )
    account_number = _resolve_field(
        fields_flat,
        "account_number_mask",
        "Account number",
        "acct_number_mask",
        "acct_number",
        "account_number_display",
    )
    account_type = _resolve_field(
        fields_flat,
        "account_type",
        "Account type",
        "furnisher_category",
    )
    status = _resolve_field(fields_flat, "status_current", "Status", "status")
    primary_issue = _resolve_primary_issue(tags)

    account_payload: dict[str, Any] = {
        "key": source.idx_key,
    }
    if source.numeric_id is not None:
        account_payload["id"] = source.numeric_id
    if furnisher is not None:
        account_payload["furnisher"] = furnisher
    if account_number is not None:
        account_payload["account_number"] = account_number
    if account_type is not None:
        account_payload["type"] = account_type
    if status is not None:
        account_payload["status"] = status
    if primary_issue is not None:
        account_payload["primary_issue"] = primary_issue

    return account_payload


def _build_bureau_summary(bureaus: Mapping[str, Any]) -> Mapping[str, Any]:
    per_bureau: dict[str, Mapping[str, Any]] = {}
    for bureau in _BUREAU_ORDER:
        payload = bureaus.get(bureau) if isinstance(bureaus, Mapping) else None
        if isinstance(payload, Mapping):
            per_bureau[bureau] = dict(payload)

    opened_values = _collect_bureau_values(per_bureau, ["opened", "date_opened", "open_date"])
    last_payment_values = _collect_bureau_values(
        per_bureau, ["last_payment", "last_payment_date", "date_of_last_payment"]
    )
    dofd_values = _collect_bureau_values(
        per_bureau,
        ["dofd", "date_of_first_delinquency", "date_of_first_default"],
    )
    balance_values = _collect_bureau_values(
        per_bureau,
        ["balance", "balance_owed", "current_balance"],
    )
    high_balance_values = _collect_bureau_values(
        per_bureau,
        ["high_balance", "high_credit", "highest_balance"],
    )
    limit_values = _collect_bureau_values(
        per_bureau,
        ["limit", "credit_limit", "credit_line"],
    )

    remarks_list: list[str] = []
    for bureau_payload in per_bureau.values():
        raw = bureau_payload.get("remarks")
        remarks = _coerce_remarks(raw)
        for remark in remarks:
            if remark not in remarks_list:
                remarks_list.append(remark)

    summary: dict[str, Any] = {
        "per_bureau": per_bureau,
    }

    opened = _majority(opened_values)
    if opened is not None:
        summary["opened"] = opened
    last_payment = _majority(last_payment_values)
    if last_payment is not None:
        summary["last_payment"] = last_payment
    dofd = _majority(dofd_values)
    if dofd is not None:
        summary["dofd"] = dofd

    balance = _aggregate_numeric(balance_values)
    if balance is not None:
        summary["balance"] = balance
    high_balance = _aggregate_numeric(high_balance_values)
    if high_balance is not None:
        summary["high_balance"] = high_balance
    limit = _aggregate_numeric(limit_values)
    if limit is not None:
        summary["limit"] = limit

    if remarks_list:
        summary["remarks"] = remarks_list

    return summary


def _collect_bureau_values(
    per_bureau: Mapping[str, Mapping[str, Any]], keys: Sequence[str]
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for bureau in _BUREAU_ORDER:
        payload = per_bureau.get(bureau)
        if not isinstance(payload, Mapping):
            continue
        value = _resolve_from_mapping(payload, keys)
        if value is None:
            continue
        values[bureau] = value
    return values


def _build_index_payload(
    packs: Iterable[tuple[str, Path, Mapping[str, Any]]],
    run_dir: Path,
) -> Mapping[str, Any]:
    entries = []
    for key, pack_path, pack_payload in packs:
        relative = pack_path.relative_to(run_dir).as_posix()
        entry = {
            "account_id": key,
            "path": relative,
        }
        account_info = pack_payload.get("account")
        if isinstance(account_info, Mapping):
            account_number = account_info.get("account_number")
            if account_number:
                entry["account_number"] = account_number
            furnisher = account_info.get("furnisher")
            if furnisher:
                entry["furnisher"] = furnisher
        entries.append(entry)

    entries.sort(key=lambda item: item["account_id"])

    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    return {
        "generated_at": timestamp,
        "packs_count": len(entries),
        "packs": entries,
    }


def _resolve_field(fields_flat: Mapping[str, Any], *keys: str) -> str | None:
    if not isinstance(fields_flat, Mapping):
        return None
    for key in keys:
        value = fields_flat.get(key)
        extracted = _extract_field_value(value)
        if extracted is not None:
            return extracted
    return None


def _extract_field_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for nested_key in ("value", "display", "normalized", "text"):
            extracted = _extract_field_value(value.get(nested_key))
            if extracted is not None:
                return extracted
        for nested_key in ("mask", "masked", "masked_value"):
            extracted = _extract_field_value(value.get(nested_key))
            if extracted is not None:
                return extracted
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for entry in value:
            extracted = _extract_field_value(entry)
            if extracted is not None:
                return extracted
        return None
    if _is_missing_text(value):
        return None
    return _normalize_text(value)


def _resolve_primary_issue(tags: Mapping[str, Any] | Sequence[Any]) -> str | None:
    if isinstance(tags, Mapping):
        primary = tags.get("primary_issue")
        normalized = _normalize_text(primary)
        if normalized:
            return normalized

    issues: list[str] = []
    if isinstance(tags, Sequence):
        for entry in tags:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("kind") == "issue":
                normalized = _normalize_text(entry.get("type") or entry.get("tag"))
                if normalized:
                    issues.append(normalized)
            else:
                normalized = _normalize_text(entry.get("primary_issue"))
                if normalized:
                    issues.append(normalized)

    priority = [
        "collection",
        "chargeoff",
        "late_90",
        "medical",
        "bankruptcy",
    ]
    for candidate in priority:
        if candidate in issues:
            return candidate
    return issues[0] if issues else None


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return None
        if trimmed in _TEXT_SENTINELS:
            return None
        return trimmed
    return str(value).strip() or None


def _resolve_from_mapping(payload: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key not in payload:
            continue
        value = payload[key]
        if _is_missing_text(value):
            continue
        return value
    return None


def _is_missing_text(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in _TEXT_SENTINELS
    return False


def _majority(values: Mapping[str, Any]) -> Any | None:
    if not values:
        return None
    counter = Counter(values.values())
    most_common = counter.most_common()
    if not most_common:
        return None
    top_count = most_common[0][1]
    candidates = {value for value, count in most_common if count == top_count}
    if len(candidates) == 1:
        return next(iter(candidates))
    for bureau in _BUREAU_ORDER:
        value = values.get(bureau)
        if value in candidates:
            return value
    return next(iter(candidates))


def _aggregate_numeric(values: Mapping[str, Any]) -> Any | None:
    numeric_entries: list[tuple[float, Any]] = []
    for value in values.values():
        number = _coerce_number(value)
        if number is not None:
            numeric_entries.append((number, value))
    if numeric_entries:
        numeric_entries.sort(key=lambda item: item[0], reverse=True)
        top_value = numeric_entries[0][0]
        if float(top_value).is_integer():
            return int(top_value)
        return top_value
    return _majority(values)


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^0-9.+-]", "", value)
        if not cleaned or cleaned in {"-", "+", "."}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _coerce_remarks(raw: Any) -> list[str]:
    remarks: list[str] = []
    if isinstance(raw, str):
        normalized = _normalize_text(raw)
        if normalized:
            remarks.append(normalized)
    elif isinstance(raw, Sequence):
        for entry in raw:
            normalized = _normalize_text(entry)
            if normalized:
                remarks.append(normalized)
    return remarks


def _default_claims_menu() -> list[Mapping[str, Any]]:
    return [
        {"claim": "not_mine", "label": "This is not my account"},
        {"claim": "paid_in_full", "label": "I paid this account"},
        {"claim": "wrong_dofd", "label": "Date of first delinquency is wrong"},
    ]


__all__ = ["build_review_packs", "get_first"]

