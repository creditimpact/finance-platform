import json

from backend.validation.prompt_templates import render_validation_prompt


def test_render_validation_prompt_includes_required_sections() -> None:
    finding = {
        "field": "account_type",
        "reason_code": "C4_TWO_MATCH_ONE_DIFF",
        "reason_label": "Account type mismatch",
        "bureaus": {
            "equifax": {"normalized": "revolving", "raw": "Revolving"},
            "experian": {"normalized": "installment", "raw": "Installment"},
        },
    }

    system_prompt, user_prompt = render_validation_prompt(
        sid="S123",
        reason_code="C4_TWO_MATCH_ONE_DIFF",
        reason_label="Account type mismatch",
        documents=["statement", "id copy"],
        finding=finding,
    )

    assert "Project: credit-analyzer" in system_prompt
    assert "Assume the consumer claims the most favorable version is accurate" in system_prompt
    assert "Decision outcomes:" in user_prompt
    assert "Evaluation guidance:" in user_prompt
    assert "Hard constraints:" in user_prompt
    assert "Field finding (verbatim JSON):" in user_prompt
    assert "C4_TWO_MATCH_ONE_DIFF" in user_prompt
    assert "documents_required" in user_prompt
    assert "{{documents | join(\", \")}}" in user_prompt

    finding_json = json.dumps(finding, ensure_ascii=False, sort_keys=True)
    assert finding_json in user_prompt


