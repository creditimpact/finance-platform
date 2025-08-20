from __future__ import annotations

"""Simple finite-state machine for account planning.

This module defines allowed transitions for ``AccountState`` records and
imposes SLA gates between steps.  ``evaluate_state`` returns a tuple of
``(allowed_tags, next_eligible_at)`` describing the actions that may be
performed now and, if no action is currently permitted, the next time the
account becomes eligible.
"""

from dataclasses import asdict
from datetime import datetime, timedelta
from typing import List, Tuple

from backend.core.models import AccountState, AccountStatus

# Number of days that must elapse before the next action is allowed after
# letters are sent to the bureaus.
_DEFAULT_SLA_DAYS = 30


def _serialize_dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _deserialize_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def evaluate_state(state: AccountState, now: datetime | None = None) -> Tuple[List[str], datetime | None]:
    """Evaluate the state machine for a single account.

    Args:
        state: The account state to evaluate.
        now:   Optional override for the current time.

    Returns:
        A tuple ``(allowed_tags, next_eligible_at)``. ``allowed_tags`` is a
        list of planner tags that may be executed immediately.  If the list is
        empty, ``next_eligible_at`` contains the datetime when the account will
        become eligible for the next step.
    """

    now = now or datetime.utcnow()
    if state.status == AccountStatus.PLANNED:
        # Initial cycle – we can send the first dispute letter immediately.
        return ["dispute"], None

    if state.status == AccountStatus.SENT:
        # We must wait for the SLA period before allowing a follow-up.
        if state.last_sent_at:
            eligible_at = state.last_sent_at + timedelta(days=_DEFAULT_SLA_DAYS)
            if now >= eligible_at:
                return ["followup"], None
            return [], eligible_at
        return ["followup"], None

    # Any terminal state – no further actions are allowed.
    return [], None


def dump_state(state: AccountState) -> dict:
    """Serialize ``AccountState`` for storage in the session manager."""

    data = asdict(state)
    data["last_sent_at"] = _serialize_dt(state.last_sent_at)
    data["next_eligible_at"] = _serialize_dt(state.next_eligible_at)
    for hist in data.get("history", []):
        hist["timestamp"] = _serialize_dt(hist.get("timestamp"))
    return data


def load_state(data: dict) -> AccountState:
    """Deserialize an ``AccountState`` from session data."""

    data = dict(data)
    if isinstance(data.get("last_sent_at"), str):
        data["last_sent_at"] = _deserialize_dt(data["last_sent_at"])
    if isinstance(data.get("next_eligible_at"), str):
        data["next_eligible_at"] = _deserialize_dt(data["next_eligible_at"])
    hist = []
    for item in data.get("history", []):
        ts = item.get("timestamp")
        if isinstance(ts, str):
            item["timestamp"] = _deserialize_dt(ts)
        hist.append(item)
    data["history"] = hist
    return AccountState(**data)
