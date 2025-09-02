"""Per-bureau account block parser."""

from __future__ import annotations

from typing import Dict, List, Tuple

from backend.core.case_store.api import upsert_account_fields
from backend.core.metrics.field_coverage import (
    emit_account_field_coverage,
    emit_session_field_coverage_summary,
)

from .tokens import ACCOUNT_FIELD_MAP, ACCOUNT_RE, parse_amount, parse_date


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


def _parse_block(block: List[str]) -> Tuple[str, Dict[str, object]]:
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
    return account_id, fields


def extract(
    lines: List[str], *, session_id: str, bureau: str
) -> List[Dict[str, object]]:
    """Extract accounts from ``lines`` and write to Case Store."""

    blocks = _split_blocks(lines)
    results: List[Dict[str, object]] = []
    for block in blocks:
        account_id, fields = _parse_block(block)
        upsert_account_fields(
            session_id=session_id, account_id=account_id, bureau=bureau, fields=fields
        )
        emit_account_field_coverage(
            session_id=session_id,
            account_id=account_id,
            bureau=bureau,
            fields=fields,
        )
        results.append({"account_id": account_id, "fields": fields})
    emit_session_field_coverage_summary(session_id=session_id)
    return results
