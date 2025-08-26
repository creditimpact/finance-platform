from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping

import backend.config as config
from backend.core.ai.adjudicator_client import call_adjudicator
from backend.core.ai.models import AIAdjudicateRequest
from backend.core.case_store.api import (
    append_artifact,
    get_account_fields,
    list_accounts,
)
from backend.core.case_store.redaction import redact_for_ai
from backend.core.case_store.telemetry import emit, timed
from backend.core.logic.report_analysis.candidate_logger import log_stageA_candidates

logger = logging.getLogger(__name__)

# Field groups used for constructing problem reasons.  These are a subset of
# the fields fetched for Stage A; ``STAGEA_REQUIRED_FIELDS`` enumerates the
# complete minimal field set we request from Case Store.
EVIDENCE_FIELDS_NUMERIC = (
    "past_due_amount",
    "balance_owed",
    "credit_limit",
)
EVIDENCE_FIELDS_STATUS = ("payment_status", "account_status")
EVIDENCE_FIELDS_HISTORY = ("two_year_payment_history", "days_late_7y")

# Explicit list of all fields Stage A fetches from Case Store.  Keeping this
# centralized avoids drifting field usage between the detector and orchestrator.
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

NEUTRAL_TIER = "none"


def neutral_stageA_decision(debug: dict | None = None) -> dict:
    return {
        "primary_issue": "unknown",
        "issue_types": [],
        "problem_reasons": [],
        "decision_source": "rules",
        "confidence": 0.0,
        "tier": NEUTRAL_TIER,
        "debug": debug or {},
    }


def adopt_or_fallback(ai_resp: dict | None, min_conf: float) -> dict:
    """Return an AI decision if it meets quality gates, otherwise neutral."""
    if (
        ai_resp
        and ai_resp.get("primary_issue") not in {"none", "unknown", None}
        and ai_resp.get("confidence", 0.0) >= min_conf
    ):
        return {
            "primary_issue": ai_resp.get("primary_issue", "unknown"),
            "issue_types": ai_resp.get("issue_types", []),
            "problem_reasons": ai_resp.get("problem_reasons", []),
            "decision_source": "ai",
            "confidence": float(ai_resp.get("confidence", 0.0)),
            "tier": ai_resp.get("tier", NEUTRAL_TIER),
            "debug": {"fields_used": ai_resp.get("fields_used", [])},
        }
    return neutral_stageA_decision()


def evaluate_with_optional_ai(
    session_id: str,
    account_id: str,
    case_fields: dict,
    doc_fingerprint: str,
    account_fingerprint: str,
) -> dict:
    """Attempt AI adjudication and fall back to neutral decision."""

    if not config.ENABLE_AI_ADJUDICATOR:
        return neutral_stageA_decision(debug={"source": "rules_v1"})

    ai_fields = redact_for_ai({"fields": case_fields})["fields"]
    req = AIAdjudicateRequest(
        doc_fingerprint=doc_fingerprint or "",
        account_fingerprint=account_fingerprint or "",
        hierarchy_version=config.AI_HIERARCHY_VERSION,
        fields=ai_fields,
    )
    resp = call_adjudicator(None, req)
    resp_dict = None
    if resp:
        resp_dict = {
            "primary_issue": resp.primary_issue,
            "tier": resp.tier,
            "confidence": float(resp.confidence),
            "problem_reasons": resp.problem_reasons,
            "fields_used": resp.fields_used,
        }
    return adopt_or_fallback(resp_dict, config.AI_MIN_CONFIDENCE)


def _format_amount(v) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def _extract_late_counts(history) -> dict:
    codes: List[str] = []
    if history is None:
        return {}
    if isinstance(history, str):
        codes = [c.strip() for c in history.split(",")]
    elif isinstance(history, list):
        codes = [str(c).strip() for c in history]
    buckets = {"30": 0, "60": 0, "90": 0, "120": 0}
    for c in codes:
        if c in buckets:
            buckets[c] += 1
        elif c.endswith("D") and c[:-1] in buckets:
            buckets[c[:-1]] += 1
        elif c.endswith("+") and c[:-1] in buckets:
            buckets[c[:-1]] += 1
    return {k: v for k, v in buckets.items() if v > 0}


def build_problem_reasons(fields: dict) -> List[str]:
    reasons: List[str] = []
    if fields.get("past_due_amount", 0):
        reasons.append(f"past_due_amount: {_format_amount(fields['past_due_amount'])}")
    for fname in EVIDENCE_FIELDS_STATUS:
        if fields.get(fname):
            reasons.append(f"status_present: {fname}")
    for hname in EVIDENCE_FIELDS_HISTORY:
        counts = _extract_late_counts(fields.get(hname))
        if counts:
            bits = [f"{v}×{k}" for k, v in counts.items()]
            reasons.append(f"late: {','.join(bits)}")
    return reasons


def evaluate_account_problem(acct: Dict[str, Any]) -> Dict[str, Any]:
    reasons = build_problem_reasons(acct)
    signals: List[Any] = []
    if acct.get("past_due_amount", 0):
        signals.append("past_due_amount")
    for fname in EVIDENCE_FIELDS_STATUS:
        if acct.get(fname):
            signals.append(f"status_present:{fname}")
    for hname in EVIDENCE_FIELDS_HISTORY:
        counts = _extract_late_counts(acct.get(hname))
        if counts:
            signals.append({hname: counts})
    decision = neutral_stageA_decision(debug={"signals": signals})
    decision["problem_reasons"] = reasons
    acct.update({k: v for k, v in decision.items() if k != "debug"})
    acct["debug"] = decision["debug"]
    acct["_detector_is_problem"] = bool(reasons)
    return decision


def run_stage_a(
    session_id: str,
    legacy_accounts: List[Mapping[str, Any]] | None = None,
) -> None:
    legacy_map = {str(a.get("account_id")): dict(a) for a in legacy_accounts or []}

    if not config.ENABLE_CASESTORE_STAGEA:
        for acc in legacy_accounts or []:
            evaluate_account_problem(acc)
        return

    try:
        account_ids = list_accounts(session_id)  # type: ignore[operator]
    except Exception:
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

            bureau = str(fields.get("bureau") or acc_id.split("_")[-1])
            if config.ENABLE_CANDIDATE_TOKEN_LOGGER:
                try:
                    log_stageA_candidates(
                        session_id,
                        acc_id,
                        bureau,
                        "pre",
                        dict(fields),
                        decision={},
                        meta={"source": "stageA"},
                    )
                except Exception:
                    logger.debug(
                        "candidate_tokens_log_failed session=%s account=%s phase=pre",
                        session_id,
                        acc_id,
                        exc_info=True,
                    )

            rules_verdict = evaluate_account_problem(dict(fields))
            ai_verdict = evaluate_with_optional_ai(
                session_id,
                acc_id,
                dict(fields),
                str(fields.get("doc_fingerprint") or ""),
                str(fields.get("account_fingerprint") or ""),
            )
            verdict = (
                ai_verdict
                if ai_verdict.get("decision_source") == "ai"
                else rules_verdict
            )

            if config.ENABLE_CANDIDATE_TOKEN_LOGGER:
                try:
                    log_stageA_candidates(
                        session_id,
                        acc_id,
                        bureau,
                        "post",
                        dict(fields),
                        verdict,
                        meta={"source": "stageA"},
                    )
                except Exception:
                    logger.debug(
                        "candidate_tokens_log_failed session=%s account=%s phase=post",
                        session_id,
                        acc_id,
                        exc_info=True,
                    )
            debug_data = dict(verdict.get("debug", {}))
            debug_data["source"] = "casestore-stageA"
            payload = {
                "primary_issue": verdict.get("primary_issue", "unknown"),
                "issue_types": verdict.get("issue_types", []),
                "problem_reasons": verdict.get("problem_reasons", []),
                "confidence": verdict.get("confidence", 0.0),
                "tier": str(verdict.get("tier", NEUTRAL_TIER)),
                "decision_source": verdict.get("decision_source", "rules"),
                "debug": debug_data,
            }
            try:
                append_artifact(  # type: ignore[operator]
                    session_id,
                    acc_id,
                    "stageA_detection",
                    payload,
                    attach_provenance={
                        "module": "problem_detection",
                        "algo": "rules_v1",
                    },
                )
            except Exception:
                logger.warning(
                    "stageA_append_failed session=%s account=%s", session_id, acc_id
                )
                emit(
                    "stageA_append_failed",
                    session_id=session_id,
                    account_id=acc_id,
                )
                continue

            if config.CASESTORE_STAGEA_LOG_PARITY:
                legacy_decision = evaluate_account_problem(
                    dict(legacy_map.get(acc_id, {}))
                )
                same_primary = legacy_decision.get("primary_issue") == verdict.get(
                    "primary_issue"
                )
                same_tier = legacy_decision.get("tier") == verdict.get("tier")
                reasons_diff = len(
                    set(legacy_decision.get("problem_reasons", []))
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
