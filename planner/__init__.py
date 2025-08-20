"""Planner entry points."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List

from backend.api.session_manager import get_session, update_session
from backend.audit.audit import emit_event
from backend.core.models import AccountState, AccountStatus

from .state_machine import dump_state, evaluate_state, load_state


def _ensure_account_states(
    session: dict, stored_states: Dict[str, dict]
) -> Dict[str, dict]:
    """Ensure every account in the current strategy has a tracked state."""

    strategy = session.get("strategy", {}) or {}
    accounts = strategy.get("accounts", [])
    for acc in accounts:
        acc_id = str(acc.get("account_id") or "")
        if acc_id and acc_id not in stored_states:
            state = AccountState(
                account_id=acc_id,
                current_cycle=0,
                current_step=0,
                status=AccountStatus.PLANNED,
            )
            stored_states[acc_id] = dump_state(state)
    return stored_states


def plan_next_step(
    session: dict, action_tags: Iterable[str], now: datetime | None = None
) -> List[str]:
    """Evaluate the FSM for all accounts and persist results.

    The function loads persisted ``AccountState`` objects for the provided
    ``session``, evaluates them via the finite-state machine and stores the
    updated state back to the session manager *before* any tactical side
    effects occur.  ``action_tags`` contains the Stage 2.5 tags that the
    strategist proposed for this run; the planner intersects these with the
    state machine's allowed tags to enforce cycle/SLA restrictions.

    Args:
        session: A mapping containing at least ``session_id`` and ``strategy``.
        action_tags: Iterable of action tags proposed by the strategist.

    Returns:
        A sorted list of planner-approved tags for the current step.
    """

    session_id = session.get("session_id")
    if not session_id:
        return []

    stored = get_session(session_id) or {}
    states_data: Dict[str, dict] = stored.get("account_states", {}) or {}
    states_data = _ensure_account_states(session, states_data)

    allowed: List[str] = []
    now = now or datetime.utcnow()
    for acc_id, data in states_data.items():
        state = load_state(data)
        if state.next_eligible_at and now < state.next_eligible_at:
            states_data[acc_id] = dump_state(state)
            continue
        tags, next_eligible_at = evaluate_state(state, now=now)
        state.next_eligible_at = next_eligible_at
        states_data[acc_id] = dump_state(state)
        allowed.extend(tags)

    action_set = {t for t in action_tags if t}
    if action_set:
        allowed = [t for t in allowed if t in action_set]

    update_session(session_id, account_states=states_data)
    return sorted(set(allowed))


def record_send(
    session: dict,
    account_ids: Iterable[str],
    now: datetime | None = None,
    sla_days: int = 30,
) -> None:
    """Record that letters were sent for the given accounts."""

    session_id = session.get("session_id")
    if not session_id:
        return

    stored = get_session(session_id) or {}
    states_data: Dict[str, dict] = stored.get("account_states", {}) or {}
    if not states_data:
        return

    now = now or datetime.utcnow()
    for acc_id in account_ids:
        data = states_data.get(str(acc_id))
        if not data:
            continue
        state = load_state(data)
        state.last_sent_at = now
        state.next_eligible_at = now + timedelta(days=sla_days)
        state.transition(AccountStatus.SENT, actor="planner")
        state.current_step += 1
        emit_event(
            "audit.planner_transition",
            {
                "account_id": str(acc_id),
                "cycle": state.current_cycle,
                "step": state.current_step,
                "reason": "letters_sent",
            },
        )
        states_data[str(acc_id)] = dump_state(state)

    update_session(session_id, account_states=states_data)
