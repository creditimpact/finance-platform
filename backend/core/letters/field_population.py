"""Helpers to populate missing fields prior to template validation."""

from __future__ import annotations

from typing import Any, Mapping

from backend.core.logic.letters.utils import populate_required_fields
from fields.populate_account_number_masked import populate_account_number_masked
from fields.populate_address import populate_address
from fields.populate_amount import populate_amount
from fields.populate_creditor_name import populate_creditor_name
from fields.populate_days_since_cra_result import populate_days_since_cra_result
from fields.populate_dob import populate_dob
from fields.populate_inquiry_creditor_name import populate_inquiry_creditor_name
from fields.populate_inquiry_date import populate_inquiry_date
from fields.populate_medical_status import populate_medical_status
from fields.populate_name import populate_name
from fields.populate_ssn_masked import populate_ssn_masked


def apply_field_fillers(
    ctx: dict,
    *,
    strategy: Mapping[str, Any] | None = None,
    profile: Mapping[str, Any] | None = None,
    corrections: Mapping[str, Any] | None = None,
) -> None:
    """Populate ``ctx`` using available field fillers.

    Parameters
    ----------
    ctx:
        Context dictionary mutated in-place.
    strategy:
        Optional per-account strategy data used by ``populate_required_fields``.
    profile:
        Optional client profile supplying PII fields.
    corrections:
        Optional corrections overriding profile values.
    """

    tri_merge = ctx.get("tri_merge") or {}
    inquiry = ctx.get("inquiry_evidence") or ctx.get("inquiry") or {}
    medical = ctx.get("medical_evidence") or ctx.get("medical") or {}
    outcome = ctx.get("cra_outcome") or ctx.get("outcome") or {}
    profile = profile or ctx.get("profile") or ctx.get("client") or {}
    corrections = corrections or ctx.get("corrections") or {}

    # Client/profile fields -----------------------------------------------------
    populate_name(ctx, profile, corrections)
    populate_address(ctx, profile, corrections)
    populate_dob(ctx, profile, corrections)
    populate_ssn_masked(ctx, profile, corrections)

    # Evidence-driven account fields -------------------------------------------
    if tri_merge.get("name") and not ctx.get("name"):
        ctx["name"] = tri_merge["name"]
    populate_creditor_name(ctx, tri_merge)
    populate_account_number_masked(ctx, tri_merge)
    populate_days_since_cra_result(ctx, outcome)
    populate_inquiry_creditor_name(ctx, inquiry)
    populate_inquiry_date(ctx, inquiry)
    populate_amount(ctx, medical)
    populate_medical_status(ctx, medical)

    # Strategy provided fields -------------------------------------------------
    populate_required_fields(ctx, strategy)


__all__ = ["apply_field_fillers"]
