from backend.core.logic.report_analysis.report_parsing import parse_account_block
from backend.core.materialize.account_materializer import _get_scalar


def test_get_scalar_prefers_normalized():
    assert _get_scalar({"raw": "10", "normalized": 10}) == 10
    assert _get_scalar({"raw": "10", "normalized": None}) == "10"
    assert _get_scalar("x") == "x"


def test_by_bureau_simple_flattens_cells():
    maps = {
        "transunion": {"credit_limit": {"raw": "100", "normalized": 100.0, "provenance": "aligned"}},
        "experian": {"credit_limit": {"raw": "200", "normalized": 200.0, "provenance": "aligned"}},
        "equifax": {"credit_limit": {"raw": "N/A", "normalized": None, "provenance": "aligned"}},
    }
    simple = {b: {k: _get_scalar(v) for k, v in bm.items()} for b, bm in maps.items()}
    assert simple["transunion"]["credit_limit"] == 100.0
    assert simple["experian"]["credit_limit"] == 200.0
    assert simple["equifax"]["credit_limit"] == "N/A"
