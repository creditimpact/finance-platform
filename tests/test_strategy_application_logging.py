import copy
from backend.core.logic.letters import letter_generator


def test_apply_strategy_fields_logs(monkeypatch):
    events = []
    monkeypatch.setattr(
        "backend.core.logic.letters.letter_generator.emit_event",
        lambda e, p: events.append((e, p)),
    )
    bureau = {
        "B": {
            "accounts": [
                {
                    "name": "Creditor",
                    "account_number": "1234",
                    "account_id": "1",
                    "action_tag": "dispute",
                }
            ]
        }
    }
    strategy_acc = [
        {
            "name": "Creditor",
            "account_number": "1234",
            "action_tag": "fraud",
            "policy_override_reason": "test_reason",
        }
    ]
    # Use copy to avoid mutation affecting assertions
    letter_generator._apply_strategy_fields(copy.deepcopy(bureau), strategy_acc)
    assert any(
        e == "strategy_applied"
        and p["action_tag_before"] == "dispute"
        and p["action_tag_after"] == "dispute"
        and p["override_reason"] == "test_reason"
        and p["strategy_applied"]
        for e, p in events
    )
