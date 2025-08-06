import json
import sys
from pathlib import Path
import importlib

sys.path.append(str(Path(__file__).resolve().parents[1]))

from session_manager import get_session


def test_explanations_endpoint_stores_raw_and_structured(monkeypatch):
    import pdfkit

    def fake_config(*args, **kwargs):
        class Dummy:
            pass
        return Dummy()

    monkeypatch.setattr("pdfkit.configuration", fake_config)

    app_module = importlib.import_module("app")
    app = app_module.app

    def fake_extract(text, ctx):
        return {
            "account_id": ctx["account_id"],
            "dispute_type": ctx["dispute_type"],
            "facts_summary": "summary",
            "claimed_errors": [],
            "dates": {},
            "evidence": [],
            "risk_flags": {},
        }

    monkeypatch.setattr(app_module, "extract_structured", fake_extract)

    session_id = "sess-expl"
    payload = {
        "session_id": session_id,
        "explanations": [
            {"account_id": "1", "dispute_type": "late", "text": "I was late"}
        ],
    }

    with app.test_client() as client:
        res = client.post("/api/explanations", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"

        res = client.get(f"/api/summaries/{session_id}")
        summary_data = res.get_json()

    session = get_session(session_id)
    assert session["raw_explanations"][0]["text"] == "I was late"
    assert session["structured_summaries"][0]["facts_summary"] == "summary"
    text = json.dumps(summary_data)
    assert "I was late" not in text
