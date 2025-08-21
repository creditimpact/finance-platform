"""Compute days_since_cra_result from an outcome timestamp."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping


def populate_days_since_cra_result(
    ctx: dict,
    outcome: Mapping[str, object] | None = None,
    *,
    now: datetime | None = None,
) -> None:
    """Populate ``days_since_cra_result`` on ``ctx`` if missing.

    ``outcome`` may provide a ``timestamp`` field representing the last CRA
    result date. If ``timestamp`` is a string, it must be ISO formatted.
    """

    if ctx.get("days_since_cra_result") is not None:
        return

    if not outcome:
        return

    ts = outcome.get("timestamp") or outcome.get("cra_result_at")
    if not ts:
        return

    if isinstance(ts, str):
        result_time = datetime.fromisoformat(ts)
    else:
        result_time = ts
    now = now or datetime.now(result_time.tzinfo or timezone.utc)
    ctx["days_since_cra_result"] = (now - result_time).days
