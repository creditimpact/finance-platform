import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from logic.rule_checker import check_letter
from session_manager import get_session, update_session


def test_ny_medical_clause_injected_and_logged(tmp_path):
    session_id = "sess-ny-compliance"
    text = "Please review this account.\nSincerely,\nJohn Doe"
    cleaned, violations = check_letter(
        text,
        state="NY",
        context={"debt_type": "medical", "session_id": session_id},
    )
    assert "new york financial services law" in cleaned.lower()
    assert cleaned.index("Additionally") < cleaned.index("Sincerely")
    session = get_session(session_id)
    assert session["state_compliance"]["state"] == "NY"
    assert "NY Medical Debt Clause" in session["state_compliance"]["clauses_added"]


def test_ga_prohibit_service_flags_violation():
    _, violations = check_letter("Body\nSincerely,\nName", state="GA", context={})
    assert any(v["rule_id"] == "STATE_PROHIBITED" and v["severity"] == "critical" for v in violations)


def test_unrelated_state_no_extra_text():
    text = "Investigate this account.\nSincerely,\nJane Doe"
    cleaned, _ = check_letter(text, state="TX", context={})
    assert "Additionally, pursuant" not in cleaned
