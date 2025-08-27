import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator, ValidationError

SCHEMA_DIR = Path("backend/schemas")


def _load(name: str) -> dict:
    with open(SCHEMA_DIR / name) as f:
        return json.load(f)


def _problem_account_base() -> dict:
    return {
        "account_id": "1",
        "bureau": "Equifax",
        "primary_issue": "collection",
        "tier": "Tier1",
        "problem_reasons": ["reason"],
        "confidence": 0.9,
        "decision_source": "ai",
        "fields_used": ["balance_owed"],
    }


def _ai_base() -> dict:
    return {
        "primary_issue": "collection",
        "tier": "Tier1",
        "problem_reasons": ["foo"],
        "confidence": 0.9,
        "fields_used": ["balance_owed"],
        "decision_source": "ai",
    }


def test_happy_path_valid_accounts():
    ai_schema = _load("ai_adjudication.json")
    account_schema = _load("problem_account.json")

    Draft7Validator(ai_schema).validate(
        {
            "primary_issue": "collection",
            "tier": "Tier1",
            "problem_reasons": ["foo"],
            "confidence": 0.85,
            "fields_used": ["balance_owed"],
            "decision_source": "ai",
        }
    )

    Draft7Validator(account_schema).validate(_problem_account_base())
    neutral = _problem_account_base()
    neutral.update({"account_id": "2", "tier": "none", "decision_source": "rules"})
    Draft7Validator(account_schema).validate(neutral)


def test_extra_property_fails():
    schema = _load("problem_account.json")
    obj = _problem_account_base()
    obj["extra"] = True
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(obj)


def test_wrong_bureau_fails():
    schema = _load("problem_account.json")
    obj = _problem_account_base()
    obj["bureau"] = "FakeBureau"
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(obj)


def test_confidence_out_of_range_fails():
    schema = _load("problem_account.json")
    obj = _problem_account_base()
    obj["confidence"] = 2.0
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(obj)


def test_ai_extra_property_fails():
    schema = _load("ai_adjudication.json")
    obj = _ai_base()
    obj["extra"] = True
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(obj)


def test_ai_bad_tier_enum_fails():
    schema = _load("ai_adjudication.json")
    obj = _ai_base()
    obj["tier"] = "TIER1"
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(obj)


def test_ai_bad_primary_issue_enum_fails():
    schema = _load("ai_adjudication.json")
    obj = _ai_base()
    obj["primary_issue"] = "bogus"
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(obj)


def test_ai_too_many_problem_reasons_fails():
    schema = _load("ai_adjudication.json")
    obj = _ai_base()
    obj["problem_reasons"] = ["r"] * 11
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(obj)


def test_ai_fields_used_illegal_chars_fails():
    schema = _load("ai_adjudication.json")
    obj = _ai_base()
    obj["fields_used"] = ["Payment Status"]
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(obj)


def test_ai_confidence_out_of_range_fails():
    schema = _load("ai_adjudication.json")
    obj = _ai_base()
    obj["confidence"] = 1.5
    with pytest.raises(ValidationError):
        Draft7Validator(schema).validate(obj)
