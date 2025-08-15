import json
from types import SimpleNamespace
from pathlib import Path

from backend.core.logic.report_analysis.report_prompting import analyze_bureau


class DummyAIClient:
    def __init__(self, content: str):
        self._content = content

    def chat_completion(self, **kwargs):
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], usage=None)


def test_cross_validation_corrects_fields(tmp_path: Path):
    text = (
        "Test Bank\n"
        "Account Number: ****1234\n"
        "Status: Open\n"
        "DOFD: 01/2020\n\n"
        "Sample Lender\n"
        "Inquiry Date: 03/15/2023\n"
    )

    ai_output = json.dumps(
        {
            "all_accounts": [
                {
                    "name": "Test Bank",
                    "account_number": "****9999",
                    "status": "Closed",
                    "dofd": "02/2020",
                }
            ],
            "inquiries": [
                {"creditor_name": "Sample Lender", "date": "02/01/2023"}
            ],
        }
    )
    ai_client = DummyAIClient(ai_output)

    data, err = analyze_bureau(
        text=text,
        is_identity_theft=False,
        output_json_path=tmp_path / "out.json",
        ai_client=ai_client,
        strategic_context=None,
        prompt="",
        late_summary_text="",
        inquiry_summary="",
    )

    assert err is None
    account = data["all_accounts"][0]
    assert account["account_number"] == "****1234"
    assert account["status"] == "Open"
    assert account["dofd"] == "01/2020"
    assert account["remediation_applied"] is True

    inquiry = data["inquiries"][0]
    assert inquiry["date"] == "03/15/2023"
    assert inquiry["remediation_applied"] is True
