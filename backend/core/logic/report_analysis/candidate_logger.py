"""Collect raw SmartCredit field values for dictionary building.

The logger stores unique field values across accounts and writes them to a
JSON file. This allows offline analysis to build comprehensive keyword
Dictionaries without impacting Stage A logic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Set


_FIELDS = [
    "balance_owed",
    "account_rating",
    "account_description",
    "dispute_status",
    "creditor_type",
    "account_status",
    "payment_status",
    "creditor_remarks",
    "account_type",
    "credit_limit",
    "late_payments",
    "past_due_amount",
]


def _redact(value: str) -> str:
    """Mask digits in string values to avoid logging PII."""
    if value.isdigit():
        return value
    return re.sub(r"\d", "X", value)


class CandidateTokenLogger:
    """Accumulates raw field values and persists them to disk."""

    def __init__(self) -> None:
        self._tokens: Dict[str, Set[str]] = {name: set() for name in _FIELDS}

    def collect(self, account: Dict[str, object]) -> None:
        for field in _FIELDS:
            val = account.get(field)
            if val is None or val == "":
                continue
            if field == "late_payments" and isinstance(val, dict):
                for bureau, buckets in val.items():
                    for days, count in (buckets or {}).items():
                        token = f"{bureau}:{days}:{count}"
                        self._tokens[field].add(token)
            elif isinstance(val, dict):
                for v in val.values():
                    if v:
                        s = str(v)
                        self._tokens[field].add(_redact(s))
            else:
                s = str(val)
                if isinstance(val, (int, float)) or s.isdigit():
                    self._tokens[field].add(s)
                else:
                    self._tokens[field].add(_redact(s))

    def save(self, folder: Path) -> None:
        """Write collected tokens to ``folder/candidate_tokens.json``."""

        data = {k: sorted(v) for k, v in self._tokens.items() if v}
        if not data:
            return
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "candidate_tokens.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
