from datetime import datetime, timedelta

from backend.core.models import AccountState, AccountStatus
from planner.state_machine import evaluate_state


def test_fsm_transitions_and_idempotency():
    state = AccountState(account_id="1", current_cycle=0, current_step=0, status=AccountStatus.PLANNED)
    now = datetime(2024, 1, 1)
    tags1, eligible1 = evaluate_state(state, now=now)
    tags2, eligible2 = evaluate_state(state, now=now)
    assert tags1 == ["dispute"]
    assert eligible1 is None
    assert (tags1, eligible1) == (tags2, eligible2)
    assert state.status == AccountStatus.PLANNED

    state.status = AccountStatus.SENT
    state.last_sent_at = now
    tags, eligible = evaluate_state(state, now=now + timedelta(days=10))
    assert tags == []
    assert eligible == now + timedelta(days=30)

    tags, eligible = evaluate_state(state, now=now + timedelta(days=31))
    assert tags == ["followup"]
    assert eligible is None

    state.status = AccountStatus.CRA_RESPONDED_DELETED
    tags, eligible = evaluate_state(state, now=now + timedelta(days=40))
    assert tags == []
    assert eligible is None
