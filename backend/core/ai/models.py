from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, confloat, constr

PrimaryIssue = Literal[
    "collection",
    "charge_off",
    "bankruptcy",
    "repossession",
    "foreclosure",
    "severe_delinquency",
    "derogatory",
    "high_utilization",
    "none",
    "unknown",
]

Tier = Literal["Tier1", "Tier2", "Tier3", "Tier4", "none"]


class AIAdjudicateRequest(BaseModel):
    doc_fingerprint: constr(strip_whitespace=True)
    account_fingerprint: constr(strip_whitespace=True)
    hierarchy_version: constr(strip_whitespace=True) = "v1"
    fields: Dict[str, Any]


class AIAdjudicateResponse(BaseModel):
    primary_issue: PrimaryIssue
    tier: Tier
    problem_reasons: List[str] = Field(default_factory=list)
    confidence: confloat(ge=0.0, le=1.0)
    fields_used: List[str] = Field(default_factory=list)
    decision_source: Literal["ai"] = "ai"


__all__ = [
    "PrimaryIssue",
    "Tier",
    "AIAdjudicateRequest",
    "AIAdjudicateResponse",
]
