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

    rp._inject_missing_late_accounts(result, history, raw_map)

    accounts = [BureauAccount.from_dict(a) for a in result["all_accounts"]]

    assert len(accounts) == 1
    acc = accounts[0]
    assert acc.extras["late_payments"] == history["cap_one"]
    assert acc.extras.get("source_stage") == "parser_aggregated"
    assert acc.extras.get("issue_types") == ["late_payment"]
    assert acc.status == "Delinquent"
    assert len(result.get("negative_accounts", [])) == 1
