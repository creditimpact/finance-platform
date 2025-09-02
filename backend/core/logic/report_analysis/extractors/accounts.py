"""Per-bureau account block parser."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from backend.core.case_store.api import (
    get_account_case,
    get_or_create_logical_account_id,
    upsert_account_fields,
)
from backend.core.case_store.models import AccountCase, AccountFields, Bureau
from backend.core.config.flags import FLAGS
from backend.core.orchestrators import compute_logical_account_key
from backend.core.metrics.field_coverage import (
    emit_account_field_coverage,
    emit_session_field_coverage_summary,
)

from .tokens import ACCOUNT_FIELD_MAP, ACCOUNT_RE, parse_amount, parse_date

logger = logging.getLogger(__name__)


_BUREAU_CODES = {"TransUnion": "TU", "Experian": "EX", "Equifax": "EQ"}


def _bureau_code(name: str) -> str:
    return _BUREAU_CODES.get(name, name[:2].upper())


def _split_blocks(lines: List[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if ACCOUNT_RE.search(line):
            if current:
                blocks.append(current)
            current = [line]
        else:
            if line.strip() == "" and current:
                blocks.append(current)
                current = []
            else:
                current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _parse_block(block: List[str]) -> Tuple[str, Dict[str, object], str]:
    first = block[0]
    m = ACCOUNT_RE.search(first)
    number = m.group(1) if m else ""
    account_id = (
        number[-4:] if number else f"synthetic-{hash(' '.join(block)) & 0xffff:x}"
    )
    fields: Dict[str, object] = {}
    for line in block[1:]:
        if ":" not in line:
            continue
        label, value = [p.strip() for p in line.split(":", 1)]
        key = ACCOUNT_FIELD_MAP.get(label.lower())
        if not key:
            continue
        if "amount" in key or key in {
            "high_balance",
            "balance_owed",
            "past_due_amount",
            "credit_limit",
            "payment_amount",
        }:
            fields[key] = parse_amount(value)
        elif key.endswith("date") or key in {
            "last_verified",
            "last_payment",
            "date_of_last_activity",
        }:
            fields[key] = parse_date(value) or value.strip()
        else:
            fields[key] = value.strip()
    return account_id, fields, number


def extract(
    lines: List[str], *, session_id: str, bureau: str
) -> List[Dict[str, object]]:
    """Extract accounts from ``lines`` and write to Case Store."""

    blocks = _split_blocks(lines)
    results: List[Dict[str, object]] = []
    for block in blocks:
        account_id, fields, number = _parse_block(block)
        if FLAGS.one_case_per_account_enabled:
            temp_case = AccountCase(
                bureau=Bureau.Equifax,
                fields=AccountFields(
                    account_number=number,
                    creditor_type=fields.get("creditor_type"),
                    account_type=fields.get("account_type"),
                    date_opened=fields.get("date_opened"),
                ),
            )
            logical_key = compute_logical_account_key(temp_case)
            account_id = get_or_create_logical_account_id(session_id, logical_key)
            upsert_account_fields(
                session_id=session_id,
                account_id=account_id,
                bureau=bureau,
                fields={"by_bureau": {_bureau_code(bureau): fields}},
            )
        else:
            upsert_account_fields(
                session_id=session_id,
                account_id=account_id,
                bureau=bureau,
                fields=fields,
            )
        emit_account_field_coverage(
            session_id=session_id,
            account_id=account_id,
            bureau=bureau,
            fields=fields,
        )
        if FLAGS.normalized_overlay_enabled:
            try:
                from backend.core.normalize.apply import (
                    build_normalized,
                    emit_mapping_coverage_metrics,
                    load_registry,
                )

                reg = load_registry()
                case = get_account_case(session_id, account_id)
                by_bureau = case.fields.model_dump().get("by_bureau", {})
                overlay = build_normalized(by_bureau, reg)
                upsert_account_fields(
                    session_id=session_id,
                    account_id=account_id,
                    bureau=None,
                    fields={"normalized": overlay},
                )
                emit_mapping_coverage_metrics(
                    session_id, account_id, by_bureau, reg
                )
            except Exception:
                logger.exception("normalized_overlay_failed")
        results.append({"account_id": account_id, "fields": fields})
    emit_session_field_coverage_summary(session_id=session_id)
    return results
