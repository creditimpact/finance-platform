"""Per-bureau account block parser."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from backend.core.case_store.api import (
    get_account_case,
    get_or_create_logical_account_id,
    upsert_account_fields,
)
from backend.core.config.flags import FLAGS
from backend.core.logic.report_analysis.keys import compute_logical_account_key
from backend.core.metrics.field_coverage import (
    EXPECTED_FIELDS,
    emit_account_field_coverage,
    emit_session_field_coverage_summary,
    _is_filled,
)
from backend.core.metrics import emit_metric
from backend.core.telemetry import metrics

from .tokens import ACCOUNT_FIELD_MAP, ACCOUNT_RE, parse_amount, parse_date

logger = logging.getLogger(__name__)


_BUREAU_CODES = {"TransUnion": "TU", "Experian": "EX", "Equifax": "EQ"}

_mode_emitted: set[str] = set()
_logical_ids: Dict[Tuple[str, str], str] = {}


def _dbg(msg: str, *args: object) -> None:
    if getattr(FLAGS, "CASEBUILDER_DEBUG", True):
        logger.debug("CASEBUILDER: " + msg, *args)


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
    input_blocks = 0
    upserted = 0
    dropped = {"missing_logical_key": 0, "min_fields": 0, "write_error": 0}

    if session_id not in _mode_emitted:
        emit_metric(
            "stage1.per_account_mode.enabled",
            1.0 if FLAGS.one_case_per_account_enabled else 0.0,
            session_id=session_id,
        )
        _mode_emitted.add(session_id)

    for block in blocks:
        input_blocks += 1
        metrics.increment("casebuilder.input_blocks", tags={"session_id": session_id})
        account_id, fields, number = _parse_block(block)
        issuer = (
            fields.get("creditor_type") or fields.get("account_type") or ""
        ).strip()
        last4 = number[-4:] if number else ""
        expected = EXPECTED_FIELDS.get(bureau, [])
        filled_count = sum(1 for f in expected if _is_filled(fields.get(f)))
        expected_count = len(expected)

        lk = compute_logical_account_key(
            fields.get("creditor_type"),
            last4 or None,
            fields.get("account_type"),
            fields.get("date_opened"),
        )
        if not lk:
            dropped["missing_logical_key"] += 1
            _dbg("drop reason=missing_logical_key issuer=%r last4=%r", issuer, last4)
            metrics.increment(
                "casebuilder.dropped",
                tags={"reason": "missing_logical_key", "session_id": session_id},
            )
            continue

        min_fields_threshold = getattr(FLAGS, "CASEBUILDER_MIN_FIELDS", 0)
        if min_fields_threshold and filled_count < min_fields_threshold:
            dropped["min_fields"] += 1
            _dbg(
                "drop reason=min_fields issuer=%r filled=%d/%d",
                issuer,
                filled_count,
                expected_count,
            )
            metrics.increment(
                "casebuilder.dropped",
                tags={"reason": "min_fields", "session_id": session_id},
            )
            continue
        try:
            if FLAGS.one_case_per_account_enabled:
                logical_key = lk
                account_id = get_or_create_logical_account_id(session_id, logical_key)
                previous = _logical_ids.get((session_id, logical_key))
                if previous and previous != account_id:
                    emit_metric(
                        "stage1.logical_index.collisions",
                        1.0,
                        session_id=session_id,
                        logical_key=logical_key,
                        ids=f"{previous},{account_id}",
                    )
                    logger.warning(
                        "logical_index_collision %s",
                        {
                            "session_id": session_id,
                            "logical_key": logical_key,
                            "ids": [previous, account_id],
                        },
                )
                _logical_ids[(session_id, logical_key)] = account_id
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
            upserted += 1
            metrics.increment(
                "casebuilder.upserted", tags={"session_id": session_id}
            )
        except Exception as e:  # pragma: no cover - diagnostic path
            dropped["write_error"] += 1
            logger.exception(
                "CASEBUILDER: write_error issuer=%r last4=%r err=%s",
                issuer,
                last4,
                e,
            )
            metrics.increment(
                "casebuilder.dropped",
                tags={"reason": "write_error", "session_id": session_id},
            )
            continue

        emit_metric(
            "stage1.by_bureau.present",
            1.0,
            session_id=session_id,
            account_id=account_id,
            bureau=_bureau_code(bureau),
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
                emit_mapping_coverage_metrics(session_id, account_id, by_bureau, reg)
            except Exception:
                logger.exception("normalized_overlay_failed")
        results.append({"account_id": account_id, "fields": fields})
    emit_session_field_coverage_summary(session_id=session_id)
    logger.info(
        "CASEBUILDER: summary session=%s input=%d upserted=%d dropped=%s",
        session_id,
        input_blocks,
        upserted,
        dropped,
    )
    metrics.gauge(
        "casebuilder.input_blocks.total",
        input_blocks,
        {"session_id": session_id},
    )
    metrics.gauge(
        "casebuilder.upserted.total",
        upserted,
        {"session_id": session_id},
    )
    for reason, count in dropped.items():
        metrics.gauge(
            "casebuilder.dropped.total",
            count,
            {"reason": reason, "session_id": session_id},
        )
    return results


def build_account_cases(session_id: str) -> None:
    """Build AccountCase records for ``session_id`` if needed."""

    # The extractor writes directly to Case Store during analysis, so no
    # additional work is required here. The function exists to provide an
    # explicit orchestration hook and remains idempotent.
    return None
