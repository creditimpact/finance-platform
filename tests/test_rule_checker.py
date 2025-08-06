from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from logic.rule_checker import check_letter


def test_admissions_replaced_and_ca_disclosure():
    text = "I admit this is my fault."
    cleaned, violations = check_letter(text, state="CA", context={})
    assert "I admit" not in cleaned
    assert "I dispute the accuracy of this information and request validation." in cleaned
    assert "California Credit Services Act disclosure" in cleaned
    assert any(v["rule_id"] == "RULE_NO_ADMISSION" for v in violations)


def test_pii_masked_and_violation_recorded():
    text = "My SSN is 123-45-6789"
    cleaned, violations = check_letter(text, state=None, context={})
    assert "123-45-6789" not in cleaned
    assert "[REDACTED]" in cleaned
    assert any(v["rule_id"] == "RULE_PII_LIMIT" and v["severity"] == "critical" for v in violations)


def test_state_specific_clause_appended_for_ny_medical():
    text = "This concerns a medical debt."
    cleaned, _ = check_letter(text, state="NY", context={"debt_type": "medical"})
    assert "pursuant to new york rules limiting medical debt reporting" in cleaned.lower()
