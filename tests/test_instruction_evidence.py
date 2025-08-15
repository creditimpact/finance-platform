import pytest

from backend.core.logic.rendering.instruction_data_preparation import prepare_instruction_data
from backend.core.logic.rendering.instruction_renderer import build_instruction_html
from tests.helpers.fake_ai_client import FakeAIClient


def test_instruction_includes_evidence():
    client_info = {"name": "Test"}
    bureau_data = {
        "Experian": {
            "all_accounts": [
                {"name": "Bank", "account_number": "1", "status": "Open", "bureaus": ["Experian"]}
            ]
        }
    }
    strategy = {
        "accounts": [
            {
                "name": "Bank",
                "account_number": "1",
                "action_tag": "dispute",
                "needs_evidence": ["identity_theft_affidavit"],
            }
        ]
    }
    fake = FakeAIClient()
    context, accounts = prepare_instruction_data(
        client_info,
        bureau_data,
        False,
        "2024-01-01",
        "",
        ai_client=fake,
        strategy=strategy,
    )
    html = build_instruction_html(context)
    assert "identity_theft_affidavit" in html
    assert accounts[0]["needs_evidence"] == ["identity_theft_affidavit"]
