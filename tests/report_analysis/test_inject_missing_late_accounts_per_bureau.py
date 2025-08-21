import backend.core.logic.report_analysis.report_postprocessing as rp
from backend.core.models.bureau import BureauAccount


def test_inject_missing_late_accounts_per_bureau():
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

    assert len(accounts) == 3
    assert {a.bureau for a in accounts} == {"Experian", "Equifax", "TransUnion"}
    assert all("bureaus" not in a for a in result["all_accounts"])
