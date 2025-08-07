from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable


def tally_failure_reasons(audit: Any) -> Dict[str, int]:
    """Return counts of strategist failure reasons from an audit object.

    The ``audit`` argument may be an :class:`AuditLogger` instance or a raw
    ``dict`` representing the audit data. Only entries that contain a
    ``failure_reason`` key are tallied.
    """
    if audit is None:
        return {}

    data = getattr(audit, "data", audit) or {}
    accounts: Dict[str, Iterable[Dict[str, Any]]] = data.get("accounts", {})

    counter: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for account_id, entries in accounts.items():
        for entry in entries:
            reason = entry.get("failure_reason")
            if reason and (account_id, reason) not in seen:
                counter[reason] += 1
                seen.add((account_id, reason))

    return dict(counter)
