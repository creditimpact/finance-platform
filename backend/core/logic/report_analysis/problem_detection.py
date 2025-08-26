"""Rule/AI hybrid problem detection for Stage A."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping

from backend.api.internal_ai import adjudicate as ai_adjudicate
from backend.config import AI_MIN_CONFIDENCE, ENABLE_AI_ADJUDICATOR
import backend.config as config

from .redaction import redact_account_for_ai

from backend.core.case_store.api import (
    append_artifact,
    get_account_case,
    get_account_fields,
    list_accounts,
)
from backend.core.case_store.telemetry import emit, timed

import logging

logger = logging.getLogger(__name__)

STAGEA_REQUIRED_FIELDS = [
    "balance_owed",
    "payment_status",
    "account_status",
    "credit_limit",
    "past_due_amount",
    "account_rating",
    "account_description",
    "creditor_remarks",
    "account_type",
    "creditor_type",
    "dispute_status",
    "two_year_payment_history",
    "days_late_7y",
]


def evaluate_account_problem(acct: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate an account for potential problems.

    The function applies lightweight rule-based checks and optionally consults
    the AI adjudicator.  It returns a unified decision dictionary and mutates
    ``acct`` in-place so the decision fields are available to later stages.
    """

    reasons: List[str] = []
    late = acct.get("late_payments") or {}
    for bureau, buckets in late.items():
        for days, count in (buckets or {}).items():
            try:
                c = int(count)
                d = int(days)
            except Exception:
                continue
            if c > 0:
                reasons.append(f"late_payment: {c}x{d} on {bureau}")
    if acct.get("past_due_amount") is not None:
        try:
            if float(acct["past_due_amount"]) > 0:
                reasons.append("past_due_amount")
        except Exception:
            pass

    decision: Dict[str, Any] = {
        "primary_issue": "unknown",
        "issue_types": [],
        "problem_reasons": list(reasons),
        "tier": 0,
        "confidence": 0.0,
        "decision_source": "rules",
        "adjudicator_version": "rules-v1",
        "debug": {
            "ai_latency_ms": 0,
            "ai_tokens_in": 0,
            "ai_tokens_out": 0,
            "ai_error": None,
        },
    }

    is_problem = bool(reasons)

    if ENABLE_AI_ADJUDICATOR:
        start = time.perf_counter()
        try:
            redacted = redact_account_for_ai(acct)
            ai_resp = ai_adjudicate("stageA", "v1", redacted)
            decision["debug"].update(
                {
                    "ai_latency_ms": int((time.perf_counter() - start) * 1000),
                    "ai_tokens_in": ai_resp.get("tokens_in", 0),
                    "ai_tokens_out": ai_resp.get("tokens_out", 0),
                    "ai_error": ai_resp.get("error"),
                }
            )
            if ai_resp.get("error"):
                decision["decision_source"] = "fallback_ai_error"
            elif (
                ai_resp.get("confidence", 0.0) >= AI_MIN_CONFIDENCE
                and ai_resp.get("primary_issue") != "unknown"
            ):
                decision.update(
                    {
                        "primary_issue": ai_resp.get("primary_issue", "unknown"),
                        "issue_types": ai_resp.get("issue_types", []),
                        "problem_reasons": ai_resp.get("problem_reasons", []),
                        "confidence": ai_resp.get("confidence", 0.0),
                        "tier": ai_resp.get("tier", 0),
                        "decision_source": "ai",
                        "adjudicator_version": ai_resp.get(
                            "adjudicator_version", "ai-v1"
                        ),
                    }
                )
                is_problem = True
            else:
                decision["decision_source"] = "fallback_ai_low_conf"
        except Exception as exc:  # pragma: no cover - defensive
            decision["decision_source"] = "fallback_ai_error"
            decision["debug"]["ai_error"] = str(exc)
            decision["debug"]["ai_latency_ms"] = int(
                (time.perf_counter() - start) * 1000
            )

    acct.update({k: v for k, v in decision.items() if k != "debug"})
    acct["debug"] = decision["debug"]
    acct["_detector_is_problem"] = bool(is_problem)
    return decision


def run_stage_a(
    session_id: str,
    legacy_accounts: List[Mapping[str, Any]] | None = None,
) -> None:
    """Evaluate Stage A for a session.

    When ``ENABLE_CASESTORE_STAGEA`` is true, account data is read from the
    Case Store.  Otherwise the provided ``legacy_accounts`` are evaluated in
    memory.  Results are written back to the Case Store under the
    ``stageA_detection`` namespace when the flag is enabled.
    """

    legacy_map = {str(a.get("account_id")): dict(a) for a in legacy_accounts or []}

    if not config.ENABLE_CASESTORE_STAGEA:
        for acc in legacy_accounts or []:
            evaluate_account_problem(acc)  # mutates in-place
        return

    account_ids: List[str] = []
    try:
        account_ids = list_accounts(session_id)  # type: ignore[operator]
    except Exception:  # pragma: no cover - defensive
        logger.warning("stageA_list_accounts_failed session=%s", session_id)
        return

    for acc_id in account_ids:
        with timed(
            "stageA_casestore_eval",
            session_id=session_id,
            account_id=acc_id,
            used_source="casestore",
        ):
            try:
                fields = get_account_fields(  # type: ignore[operator]
                    session_id, acc_id, STAGEA_REQUIRED_FIELDS
                )
            except Exception:
                logger.warning(
                    "stageA_missing_account session=%s account=%s", session_id, acc_id
                )
                emit(
                    "stageA_missing_account",
                    session_id=session_id,
                    account_id=acc_id,
                )
                continue

            verdict = evaluate_account_problem(dict(fields))
            payload = {
                "primary_issue": verdict.get("primary_issue", "unknown"),
                "issue_types": verdict.get("issue_types", []),
                "problem_reasons": verdict.get("problem_reasons", []),
                "confidence": verdict.get("confidence", 0.0),
                "tier": str(verdict.get("tier", "none")),
                "decision_source": verdict.get("decision_source", "rules"),
                "debug": {"source": "casestore-stageA"},
            }
            append_artifact(  # type: ignore[operator]
                session_id,
                acc_id,
                "stageA_detection",
                payload,
                attach_provenance={"module": "problem_detection", "algo": "rules_v1"},
            )

            if config.CASESTORE_STAGEA_LOG_PARITY and acc_id in legacy_map:
                legacy_verdict = evaluate_account_problem(dict(legacy_map[acc_id]))
                same_primary = (
                    legacy_verdict.get("primary_issue")
                    == verdict.get("primary_issue")
                )
                same_tier = legacy_verdict.get("tier") == verdict.get("tier")
                reasons_diff = len(
                    set(legacy_verdict.get("problem_reasons", []))
                    ^ set(verdict.get("problem_reasons", []))
                )
                logger.info(
                    "stageA_parity: session=%s account=%s same_primary=%s same_tier=%s reasons_diff=%d",
                    session_id,
                    acc_id,
                    same_primary,
                    same_tier,
                    reasons_diff,
                )

