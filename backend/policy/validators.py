from __future__ import annotations

"""Additional validations for the policy rulebook."""

from pathlib import Path
from typing import Any, Mapping, Set

import yaml

# Mismatch types produced by ``compute_mismatches`` in
# ``backend.core.logic.report_analysis.tri_merge``.
TRI_MERGE_MISMATCH_TYPES: Set[str] = {
    "presence",
    "balance",
    "status",
    "dates",
    "remarks",
    "utilization",
    "personal_info",
    "duplicate",
}


def _collect_tri_merge_fields(obj: Any, found: Set[str]) -> None:
    """Recursively collect tri-merge mismatch fields from condition trees."""
    if isinstance(obj, dict):
        field = obj.get("field")
        if isinstance(field, str) and field.startswith("tri_merge."):
            found.add(field.split(".", 1)[1])
        for value in obj.values():
            _collect_tri_merge_fields(value, found)
    elif isinstance(obj, list):
        for item in obj:
            _collect_tri_merge_fields(item, found)


def validate_tri_merge_mismatch_rules(rulebook: Mapping[str, Any] | None = None) -> None:
    """Ensure every mismatch type has a corresponding rule in the rulebook.

    If ``rulebook`` is not provided, ``backend/policy/rulebook.yaml`` is loaded.
    A ``ValueError`` is raised if any known mismatch type is missing.
    """

    if rulebook is None:
        path = Path(__file__).with_name("rulebook.yaml")
        rulebook = yaml.safe_load(path.read_text(encoding="utf-8"))

    found: Set[str] = set()
    for rule in rulebook.get("rules", []):
        when = rule.get("when") or rule.get("conditions")
        if when is not None:
            _collect_tri_merge_fields(when, found)

    missing = TRI_MERGE_MISMATCH_TYPES - found
    if missing:
        raise ValueError(
            "Missing tri-merge rules for mismatch types: " + ", ".join(sorted(missing))
        )


__all__ = ["validate_tri_merge_mismatch_rules", "TRI_MERGE_MISMATCH_TYPES"]
