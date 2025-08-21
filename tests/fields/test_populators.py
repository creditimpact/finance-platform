from datetime import datetime

from fields.populate_account_number_masked import populate_account_number_masked
from fields.populate_address import populate_address
from fields.populate_amount import populate_amount
from fields.populate_creditor_name import populate_creditor_name
from fields.populate_days_since_cra_result import populate_days_since_cra_result
from fields.populate_dob import populate_dob
from fields.populate_inquiry_creditor_name import populate_inquiry_creditor_name
from fields.populate_inquiry_date import populate_inquiry_date
from fields.populate_medical_status import populate_medical_status
from fields.populate_name import populate_name
from fields.populate_ssn_masked import populate_ssn_masked


def test_creditor_and_account_fields_idempotent():
    ctx: dict = {}
    populate_creditor_name(ctx, {"name": "ABC"})
    populate_account_number_masked(ctx, {"account_number": "****1"})
    assert ctx["creditor_name"] == "ABC"
    assert ctx["account_number_masked"] == "****1"

    populate_creditor_name(ctx, {"name": "XYZ"})
    populate_account_number_masked(ctx, {"account_number": "****2"})
    assert ctx["creditor_name"] == "ABC"
    assert ctx["account_number_masked"] == "****1"


def test_days_since_cra_result_computation():
    ctx: dict = {}
    ts = datetime(2024, 1, 1)
    populate_days_since_cra_result(ctx, {"timestamp": ts}, now=datetime(2024, 1, 31))
    assert ctx["days_since_cra_result"] == 30

    # second call does not override
    populate_days_since_cra_result(ctx, {"timestamp": ts}, now=datetime(2024, 2, 1))
    assert ctx["days_since_cra_result"] == 30


def test_pii_population():
    ctx: dict = {}
    profile = {
        "name": "Profile Name",
        "address": "123 Main St",
        "date_of_birth": "2000-01-01",
        "ssn_last4": "1234",
    }
    corrections = {"name": "Correct Name", "address": "456 Elm St"}
    populate_name(ctx, profile, corrections)
    populate_address(ctx, profile, corrections)
    populate_dob(ctx, profile, corrections)
    populate_ssn_masked(ctx, profile, corrections)
    assert ctx == {
        "name": "Correct Name",
        "address": "456 Elm St",
        "date_of_birth": "2000-01-01",
        "ssn_masked": "***-**-1234",
    }

    populate_name(ctx, {"name": "Other"}, {})
    assert ctx["name"] == "Correct Name"


def test_inquiry_and_medical_fields():
    ctx: dict = {}
    inquiry_evidence = {"name": "Inq Co", "date": "2024-02-01"}
    populate_inquiry_creditor_name(ctx, inquiry_evidence)
    populate_inquiry_date(ctx, inquiry_evidence)
    assert ctx["inquiry_creditor_name"] == "Inq Co"
    assert ctx["inquiry_date"] == "2024-02-01"

    populate_inquiry_creditor_name(ctx, {"name": "Other"})
    assert ctx["inquiry_creditor_name"] == "Inq Co"

    medical_evidence = {"amount": 100, "status": "Unpaid"}
    populate_amount(ctx, medical_evidence)
    populate_medical_status(ctx, medical_evidence)
    assert ctx["amount"] == 100
    assert ctx["medical_status"] == "Unpaid"

    populate_amount(ctx, {"amount": 200})
    assert ctx["amount"] == 100
