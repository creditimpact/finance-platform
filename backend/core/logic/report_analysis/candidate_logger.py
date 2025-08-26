"""Collect raw SmartCredit field values for dictionary building.

The logger stores unique field values across accounts and writes them to a
JSON file. This allows offline analysis to build comprehensive keyword
dictionaries without impacting Stage A logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Set


class CandidateTokenLogger:
    """Accumulates raw field values and persists them to disk."""

    def __init__(self) -> None:
        self._tokens: Dict[str, Set[str]] = {
            "account_status": set(),
            "payment_status": set(),
            "account_description": set(),
            "creditor_remarks": set(),
        }

    def collect(self, account: Dict[str, object]) -> None:
        for field in list(self._tokens.keys()):
            val = account.get(field)
            if isinstance(val, dict):
                for v in val.values():
                    if v:
                        self._tokens[field].add(str(v))
            elif val:
                self._tokens[field].add(str(val))

    def save(self, folder: Path) -> None:
        """Write collected tokens to ``folder/candidate_tokens.json``."""

        data = {k: sorted(v) for k, v in self._tokens.items() if v}
        if not data:
            return
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "candidate_tokens.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
