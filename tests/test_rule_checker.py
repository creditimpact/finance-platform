from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.core.logic.rule_checker import check_letter


def test_admissions_replaced_and_ca_disclosure():
    text = "I admit this is my fault."
    cleaned, violations = check_letter(text, state="CA", context={})
    assert "I admit" not in cleaned
    assert (
        "I dispute the accuracy of this information and request validation under applicable law."
        in cleaned
    )
    assert (
        "Under the California Credit Services Act, we are required to provide this disclosure."
        in cleaned
    )
    assert any(v["rule_id"] == "RULE_NO_ADMISSION" for v in violations)


def test_pii_masked_and_violation_recorded():
    text = "My SSN is 123-45-6789"
    cleaned, violations = check_letter(text, state=None, context={})
    assert "123-45-6789" not in cleaned
    assert "[REDACTED]" in cleaned
    assert any(
        v["rule_id"] == "RULE_PII_LIMIT" and v["severity"] == "critical"
        for v in violations
    )


def test_state_specific_clause_appended_for_ny_medical():
    text = "This concerns a medical debt."
    cleaned, _ = check_letter(text, state="NY", context={"debt_type": "medical"})
    assert "new york financial services law" in cleaned.lower()


def test_ga_service_prohibited():
    text = "Irrelevant"
    _, violations = check_letter(text, state="GA", context={})
    assert any(v["rule_id"] == "STATE_PROHIBITED" for v in violations)


def test_neutral_language_enforced():
    text = "These crooks are running a scam!"
    cleaned, violations = check_letter(text, state=None, context={})
    assert "crooks" not in cleaned.lower()
    assert "I believe there may be an error." in cleaned
    assert any(v["rule_id"] == "RULE_NEUTRAL_LANGUAGE" for v in violations)
