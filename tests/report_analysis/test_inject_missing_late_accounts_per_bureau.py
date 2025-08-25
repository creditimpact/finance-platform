import backend.core.logic.report_analysis.report_postprocessing as rp
from backend.core.models.bureau import BureauAccount


def test_inject_missing_late_accounts_aggregated():
    result = {}
    history = {
        "cap_one": {
            "Experian": {"30": 1},
            "Equifax": {"60": 2},
            "TransUnion": {"90": 3},
        }
    }
    raw_map = {"cap_one": "Cap One"}

    rp._inject_missing_late_accounts(result, history, raw_map, {})

    accounts = [BureauAccount.from_dict(a) for a in result["all_accounts"]]

    assert len(accounts) == 1
    acc = accounts[0]
    assert acc.late_payments == history["cap_one"]
    assert acc.source_stage == "parser_aggregated"
    assert acc.issue_types == ["late_payment"]
    assert acc.extras.get("status") == "Delinquent"
    assert len(result.get("negative_accounts", [])) == 1


def test_inject_missing_late_accounts_detects_charge_off():
    result = {}
    history = {"cap_one": {"Experian": {"30": 1}}}
    raw_map = {"cap_one": "Cap One"}
    grid_map = {"cap_one": {"Experian": "OK CO"}}

    rp._inject_missing_late_accounts(result, history, raw_map, grid_map)

    accounts = [BureauAccount.from_dict(a) for a in result["all_accounts"]]

    assert len(accounts) == 1
    acc = accounts[0]
    assert acc.late_payments == history["cap_one"]
    assert acc.extras.get("grid_history_raw") == grid_map["cap_one"]
    assert acc.source_stage == "parser_aggregated"
    assert acc.issue_types == ["charge_off", "late_payment"]
    assert acc.primary_issue == "charge_off"
    assert acc.extras.get("status") == "Charge Off"
