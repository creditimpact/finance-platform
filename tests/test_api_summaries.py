import sys
from pathlib import Path
import importlib

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.api.session_manager import update_session


def test_summaries_endpoint_returns_clean_data(monkeypatch):
    def fake_config(*args, **kwargs):
        class Dummy:
            pass

        return Dummy()

    monkeypatch.setattr("pdfkit.configuration", fake_config)
    app = importlib.import_module("app").create_app()

    session_id = "sess-summary"
    raw = {
        "account_id": "acc1",
        "dispute_type": "late_payment",
        "facts_summary": "bank error",
        "claimed_errors": ["error"],
        "dates": {"incident": "2024-01-01"},
        "evidence": ["doc"],
        "risk_flags": {"possible_identity_theft": False},
        "extra": "should be removed",
    }
    update_session(session_id, structured_summaries={"acc1": raw})

    with app.test_client() as client:
        res = client.get(f"/api/summaries/{session_id}")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"
        assert "extra" not in data["summaries"]["acc1"]
        assert data["summaries"]["acc1"]["account_id"] == "acc1"
