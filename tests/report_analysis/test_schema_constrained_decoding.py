from types import SimpleNamespace
from pathlib import Path

from backend.core.logic.report_analysis.report_prompting import analyze_bureau


class DummyAIClient:
    def __init__(self, content: str):
        self._content = content

    def chat_completion(self, **kwargs):
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice], usage=None)


def test_malformed_json_repair(tmp_path: Path):
    ai_client = DummyAIClient("{'personal_info_issues': [],}")
    data, err = analyze_bureau(
        text="",
        is_identity_theft=False,
        output_json_path=tmp_path / "out.json",
        ai_client=ai_client,
        strategic_context=None,
        prompt="",
        late_summary_text="",
        inquiry_summary="",
    )
    assert err is None
    assert data["personal_info_issues"] == []
    assert "negative_accounts" in data
