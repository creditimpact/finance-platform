import os

from backend.analytics.analytics_tracker import reset_counters
from backend.core.letters.router import route_accounts, select_template


def _setup_env():
    os.environ["LETTERS_ROUTER_PHASED"] = "1"
    reset_counters()


def test_route_accounts_matches_sequential(monkeypatch):
    _setup_env()
    accounts = [
        (
            "fraud_dispute",
            {
                "bureau": "experian",
                "creditor_name": "Acme",
                "account_number_masked": "1234",
                "legal_safe_summary": "ok",
                "is_identity_theft": True,
            },
            "s1",
        ),
        ("custom_letter", {"recipient": "Friend"}, "s2"),
    ]
    seq = [select_template(tag, ctx, "candidate", sid) for tag, ctx, sid in accounts]
    par = route_accounts(accounts, phase="candidate", max_workers=2)
    assert [d.template_path for d in par] == [d.template_path for d in seq]
