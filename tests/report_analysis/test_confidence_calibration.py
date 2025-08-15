import json
from types import SimpleNamespace
from pathlib import Path

import backend.core.logic.report_analysis.report_prompting as rp
from backend.core.logic.report_analysis.report_prompting import analyze_bureau


class DummyAIClient:
    def __init__(self, content: str):
        self._content = content

    def chat_completion(self, **kwargs):
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], usage=None)


def _patch_extractors(monkeypatch):
    def fake_headings(text: str):
        res = []
        if "Cap One" in text:
            res.append(("cap one", "Cap One"))
        if "Chase Bank" in text:
            res.append(("chase bank", "Chase Bank"))
        return res

    def fake_masks(text: str):
        masks = {}
        if "Cap One" in text:
            masks["cap one"] = "****1234"
        if "Chase Bank" in text:
            masks["chase bank"] = "****5678"
        return masks

    def fake_statuses(text: str):
        statuses = {}
        if "Cap One" in text:
            statuses["cap one"] = "Open"
        if "Chase Bank" in text:
            statuses["chase bank"] = "Closed"
        return statuses

    monkeypatch.setattr(rp, "extract_account_headings", fake_headings)
    monkeypatch.setattr(rp, "extract_account_number_masks", fake_masks)
    monkeypatch.setattr(rp, "extract_account_statuses", fake_statuses)
    monkeypatch.setattr(rp, "extract_dofd", lambda text: {})
    monkeypatch.setattr(rp, "extract_inquiry_dates", lambda text: {})
    monkeypatch.setattr(rp, "normalize_creditor_name", lambda n: (n or "").lower())


def test_confidence_easy_vs_hard(tmp_path: Path, monkeypatch):
    _patch_extractors(monkeypatch)

    text_easy = "Cap One\nAccount Number: ****1234\nStatus: Open\n"
    easy_ai = DummyAIClient(
        json.dumps(
            {
                "all_accounts": [
                    {
                        "name": "Cap One",
                        "account_number": "****1234",
                        "status": "Open",
                    }
                ],
                "inquiries": [],
            }
        )
    )
    data_easy, err_easy = analyze_bureau(
        text=text_easy,
        is_identity_theft=False,
        output_json_path=tmp_path / "easy.json",
        ai_client=easy_ai,
        strategic_context=None,
        prompt="",
        late_summary_text="",
        inquiry_summary="",
    )
    assert err_easy is None
    assert data_easy["confidence"] > 0.9
    assert data_easy["needs_human_review"] is False

    text_hard = (
        "Cap One\nAccount Number: ****1234\nStatus: Open\n"
        "Chase Bank\nAccount Number: ****5678\nStatus: Closed\n"
    )
    hard_ai = DummyAIClient(
        json.dumps(
            {
                "all_accounts": [
                    {
                        "name": "Cap One",
                        "account_number": "****9999",
                        "status": "Closed",
                    }
                ],
                "inquiries": [],
            }
        )
    )
    data_hard, err_hard = analyze_bureau(
        text=text_hard,
        is_identity_theft=False,
        output_json_path=tmp_path / "hard.json",
        ai_client=hard_ai,
        strategic_context=None,
        prompt="",
        late_summary_text="",
        inquiry_summary="",
    )
    assert err_hard is None
    assert data_hard["confidence"] < 0.7
    assert data_hard["needs_human_review"] is True
