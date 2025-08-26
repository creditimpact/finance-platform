from importlib import reload

import backend.api.internal_ai as ai
import backend.config as base_config
import backend.core.logic.report_analysis.problem_detection as pd


def _reload(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    reload(base_config)
    reload(ai)
    reload(pd)


def test_ai_high_confidence(monkeypatch):
    def fake_adjudicate(session_id, hv, account):
        return {
            "primary_issue": "collection",
            "issue_types": ["collection"],
            "problem_reasons": ["ai"],
            "confidence": 0.9,
            "tier": 1,
            "decision_source": "ai",
            "adjudicator_version": "ai-v1",
            "advice": None,
            "error": None,
        }

    _reload(monkeypatch, ENABLE_AI_ADJUDICATOR="1")
    monkeypatch.setattr(pd, "ai_adjudicate", fake_adjudicate)
    acct = {"account_status": "Open"}
    result = pd.evaluate_account_problem(acct)
    assert result["primary_issue"] == "collection"
    assert result["decision_source"] == "ai"


def test_ai_low_confidence_fallback(monkeypatch):
    def fake_adjudicate(session_id, hv, account):
        return {
            "primary_issue": "collection",
            "issue_types": ["collection"],
            "problem_reasons": ["ai"],
            "confidence": 0.2,
            "tier": 1,
            "decision_source": "ai",
            "adjudicator_version": "ai-v1",
            "advice": None,
            "error": None,
        }

    _reload(monkeypatch, ENABLE_AI_ADJUDICATOR="1", AI_MIN_CONFIDENCE="0.65")
    monkeypatch.setattr(pd, "ai_adjudicate", fake_adjudicate)
    acct = {"account_status": "Open"}
    result = pd.evaluate_account_problem(acct)
    assert result["primary_issue"] == "unknown"
    assert result["decision_source"] == "rules"
