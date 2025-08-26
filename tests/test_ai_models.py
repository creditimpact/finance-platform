from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.core.ai.models import AIAdjudicateRequest, AIAdjudicateResponse


def test_request_model_valid():
    req = AIAdjudicateRequest(
        doc_fingerprint="doc",
        account_fingerprint="acct",
        fields={"balance_owed": 100},
    )
    assert req.hierarchy_version == "v1"


def test_response_model_valid():
    resp = AIAdjudicateResponse(
        primary_issue="collection",
        tier="Tier1",
        confidence=0.9,
        problem_reasons=["reason"],
        fields_used=["balance_owed"],
    )
    assert resp.decision_source == "ai"
    assert resp.problem_reasons == ["reason"]


def test_invalid_primary_issue():
    with pytest.raises(ValidationError):
        AIAdjudicateResponse(
            primary_issue="bogus",  # type: ignore[arg-type]
            tier="Tier1",
            confidence=0.5,
        )


def test_missing_fields_in_request():
    with pytest.raises(ValidationError):
        AIAdjudicateRequest(doc_fingerprint="doc", fields={})  # type: ignore[call-arg]
