"""Internal endpoint for AI-based account adjudication."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from backend.config import ENABLE_AI_ADJUDICATOR

logger = logging.getLogger(__name__)

internal_ai_bp = Blueprint("internal_ai", __name__)


def _default_response(error: str | None = None) -> Dict[str, Any]:
    return {
        "primary_issue": "unknown",
        "issue_types": [],
        "problem_reasons": [],
        "confidence": 0.0,
        "tier": 0,
        "decision_source": "ai",
        "adjudicator_version": "ai-v1",
        "advice": None,
        "error": error,
    }


def _basic_model(account: Dict[str, Any]) -> Dict[str, Any]:
    status = str(account.get("account_status", "")).lower()
    if "collection" in status:
        return {
            "primary_issue": "collection",
            "issue_types": ["collection"],
            "problem_reasons": ["account_status:collection"],
            "confidence": 0.9,
            "tier": 1,
            "decision_source": "ai",
            "adjudicator_version": "ai-v1",
            "advice": None,
            "error": None,
        }
    return _default_response()


def adjudicate(
    session_id: str, hierarchy_version: str, account: Dict[str, Any]
) -> Dict[str, Any]:
    """Core adjudication logic used by the endpoint and Stage A."""

    start = time.time()
    if not ENABLE_AI_ADJUDICATOR:
        result = _default_response("Disabled")
    else:
        try:
            result = _basic_model(account)
        except TimeoutError:
            result = _default_response("Timeout")
        except Exception as exc:  # pragma: no cover - unexpected
            result = _default_response(str(exc))
    latency_ms = int((time.time() - start) * 1000)
    log_data = {
        "ai_call_ms": latency_ms,
        "tokens_in": 0,
        "tokens_out": 0,
        "confidence": result.get("confidence", 0.0),
        "tier": result.get("tier", 0),
        "error": result.get("error"),
    }
    logger.info("ai_adjudicate %s", json.dumps(log_data, sort_keys=True))
    return result


@internal_ai_bp.route("/internal/ai-adjudicate", methods=["POST"])
def ai_adjudicate_endpoint() -> Any:
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id", "")
    hierarchy_version = data.get("hierarchy_version", "")
    account = data.get("account") or {}
    result = adjudicate(session_id, hierarchy_version, account)
    return jsonify(result)
