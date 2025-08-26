"""Internal AI adjudication endpoints."""

from __future__ import annotations

import json
import logging
from typing import Any

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from backend.config import AI_REQUEST_TIMEOUT_S
from backend.core.ai.models import AIAdjudicateRequest, AIAdjudicateResponse
from backend.core.ai.service import run_llm_prompt

logger = logging.getLogger(__name__)

ai_bp = Blueprint("ai_internal", __name__)


@ai_bp.post("/internal/ai-adjudicate")
def ai_adjudicate() -> Any:
    """Internal-only AI adjudication endpoint."""
    try:
        raw = request.get_json(force=True) or {}
        req = AIAdjudicateRequest.model_validate(raw)
    except ValidationError:
        return jsonify({"error": "invalid_request"}), 400

    system_prompt = f"hierarchy_version={req.hierarchy_version}"
    try:
        llm_raw = run_llm_prompt(
            system_prompt,
            req.fields,
            temperature=0.0,
            timeout=AI_REQUEST_TIMEOUT_S,
        )
        data = json.loads(llm_raw)
        resp = AIAdjudicateResponse.model_validate(data)
        return jsonify(resp.model_dump())
    except TimeoutError:
        return jsonify({"error": "timeout"}), 504
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("ai_adjudicate_failure: %s", exc)
        return jsonify({"error": "bad_gateway"}), 502
