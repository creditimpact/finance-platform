"""Planner entry points."""

from __future__ import annotations

from typing import Dict, Iterable, List

from backend.api.session_manager import get_session, update_session
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


def plan_next_step(session: dict, action_tags: Iterable[str]) -> List[str]:
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
    for acc_id, data in states_data.items():
        state = load_state(data)
        tags, next_eligible_at = evaluate_state(state)
        state.next_eligible_at = next_eligible_at
        states_data[acc_id] = dump_state(state)
        allowed.extend(tags)

    action_set = {t for t in action_tags if t}
    if action_set:
        allowed = [t for t in allowed if t in action_set]

    update_session(session_id, account_states=states_data)
    return sorted(set(allowed))
