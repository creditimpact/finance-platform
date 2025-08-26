from importlib import reload

import backend.config as base_config
import backend.core.logic.report_analysis.problem_detection as pd


def _reload(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    reload(base_config)
    reload(pd)


def test_detects_numeric_and_structural(monkeypatch):
    _reload(
        monkeypatch,
        ENABLE_TIER1_KEYWORDS="0",
        ENABLE_TIER2_KEYWORDS="0",
        ENABLE_TIER3_KEYWORDS="0",
    )

    acct1 = {"bureau_statuses": {"Experian": "Collection"}}
    v1 = pd.evaluate_account_problem(acct1)
    assert v1["primary_issue"] == "collection"

    acct2 = {"late_payments": {"Experian": {"60": 1}}}
    v2 = pd.evaluate_account_problem(acct2)
    assert v2["primary_issue"] == "serious_delinquency"


def test_clean_account_not_flagged(monkeypatch):
    _reload(
        monkeypatch,
        ENABLE_TIER1_KEYWORDS="0",
        ENABLE_TIER2_KEYWORDS="0",
        ENABLE_TIER3_KEYWORDS="0",
    )
    clean = {"account_status": "Open", "payment_status": "Pays as agreed"}
    v = pd.evaluate_account_problem(clean)
    assert v["primary_issue"] == "unknown"
