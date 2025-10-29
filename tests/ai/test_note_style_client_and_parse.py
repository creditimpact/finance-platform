import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from backend.ai.note_style.parse import NoteStyleParseError, parse_note_style_response_payload
from backend.ai.note_style_ingest import ingest_note_style_result
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths
from backend.core.services.ai_client import AIClient, AIClientProtocolError
from backend.core.services.ai_config import AIConfig
from backend.util.json_tools import try_fix_to_json


def _install_openai_stub(monkeypatch: pytest.MonkeyPatch, response: Any) -> SimpleNamespace:
    holder = SimpleNamespace(instance=None)

    class _StubCompletions:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return response

    class _StubChat:
        def __init__(self) -> None:
            self.completions = _StubCompletions()

    class _StubOpenAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.chat = _StubChat()

    def _factory(*args: Any, **kwargs: Any) -> _StubOpenAI:
        instance = _StubOpenAI(*args, **kwargs)
        holder.instance = instance
        return instance

    monkeypatch.setattr("backend.core.services.ai_client.OpenAI", _factory)
    return holder


def _make_client(monkeypatch: pytest.MonkeyPatch, response: Any) -> tuple[AIClient, SimpleNamespace]:
    stub = _install_openai_stub(monkeypatch, response)
    client = AIClient(
        AIConfig(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            chat_model="gpt-4.1",
            response_model="gpt-4.1-mini",
        )
    )
    return client, stub


def _analysis_payload() -> dict[str, Any]:
    return {
        "tone": "warm",
        "context_hints": {
            "timeframe": {"month": "March", "relative": "last month"},
            "topic": "testing",
            "entities": {"creditor": "Example Bank", "amount": 125.0},
        },
        "emphasis": ["clarity", "empathy"],
        "confidence": 0.85,
        "risk_flags": ["compliance_check"],
    }


def _prepare_account(tmp_path: Path, sid: str, account_id: str, pack_payload: dict[str, Any]) -> None:
    paths = ensure_note_style_paths(tmp_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)
    account_paths.pack_file.write_text(json.dumps(pack_payload) + "\n", encoding="utf-8")


def _ingest(
    tmp_path: Path,
    *,
    sid: str,
    account_id: str,
    pack_payload: dict[str, Any],
    response_payload: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    _prepare_account(tmp_path, sid, account_id, pack_payload)

    def _complete_result(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, int], Any, bool, Any]:
        return None, {"completed": 1, "failed": 0}, None, True, None

    class _StageView(SimpleNamespace):
        is_terminal = False

    monkeypatch.setattr(
        "backend.ai.note_style_ingest.complete_note_style_result",
        _complete_result,
    )
    monkeypatch.setattr(
        "backend.ai.note_style_ingest.note_style_stage_view",
        lambda *args, **kwargs: _StageView(),
    )
    monkeypatch.setattr(
        "backend.ai.note_style_ingest.update_note_style_stage_status",
        lambda *args, **kwargs: None,
    )

    paths = ensure_note_style_paths(tmp_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    return ingest_note_style_result(
        sid=sid,
        account_id=account_id,
        runs_root=tmp_path,
        account_paths=account_paths,
        pack_payload=pack_payload,
        response_payload=response_payload,
    )


def test_chat_completion_returns_content_json_and_ingest_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    analysis = _analysis_payload()
    response_payload = analysis
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(response_payload), tool_calls=None
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    client, stub = _make_client(monkeypatch, response)
    payload = client.chat_completion(messages=[{"role": "user", "content": "Test"}])

    assert payload["mode"] == "content"
    assert payload["json"] == response_payload

    pack_payload = {"note_text": "Example note"}
    result_path = _ingest(
        tmp_path,
        sid="SID123",
        account_id="acct-1",
        pack_payload=pack_payload,
        response_payload=payload,
        monkeypatch=monkeypatch,
    )

    assert result_path.exists()

    stored = json.loads(result_path.read_text(encoding="utf-8").splitlines()[-1])
    stored_analysis = stored["analysis"]
    assert stored_analysis["tone"] == analysis["tone"]
    assert stored_analysis["context_hints"]["topic"] == analysis["context_hints"]["topic"]
    assert stored_analysis["context_hints"]["entities"]["creditor"] == analysis["context_hints"]["entities"]["creditor"]
    assert stored_analysis["risk_flags"] == analysis["risk_flags"]
    assert 0.0 <= stored_analysis["confidence"] <= 0.5
    assert stored["note_metrics"] == {"char_len": 12, "word_len": 2}

    completions = stub.instance.chat.completions
    assert completions.calls[0]["response_format"] == {"type": "json_object"}


def test_chat_completion_returns_tool_json_and_ingest_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    analysis = _analysis_payload()
    tool_arguments = json.dumps(analysis)
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(function=SimpleNamespace(arguments=tool_arguments))
                    ],
                )
            )
        ],
        usage=None,
    )

    client, _ = _make_client(monkeypatch, response)
    payload = client.chat_completion(messages=[{"role": "user", "content": "Test"}], tools=[{"type": "function"}])

    assert payload["mode"] == "tool"
    assert payload["json"] == analysis

    pack_payload = {"note_text": "Example note"}
    result_path = _ingest(
        tmp_path,
        sid="SID456",
        account_id="acct-2",
        pack_payload=pack_payload,
        response_payload=payload,
        monkeypatch=monkeypatch,
    )

    assert result_path.exists()
    stored = json.loads(result_path.read_text(encoding="utf-8").splitlines()[-1])
    stored_analysis = stored["analysis"]
    assert stored_analysis["tone"] == analysis["tone"]
    assert stored_analysis["context_hints"]["topic"] == analysis["context_hints"]["topic"]
    assert stored_analysis["context_hints"]["entities"]["creditor"] == analysis["context_hints"]["entities"]["creditor"]
    assert stored_analysis["risk_flags"] == analysis["risk_flags"]
    assert 0.0 <= stored_analysis["confidence"] <= 0.5
    assert stored["note_metrics"] == {"char_len": 12, "word_len": 2}


def test_chat_completion_handles_tool_arguments_mapping(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    analysis = _analysis_payload()
    tool_arguments = {"note": "Generated note", "analysis": analysis}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(function=SimpleNamespace(arguments=tool_arguments))
                    ],
                )
            )
        ],
        usage=None,
    )

    client, _ = _make_client(monkeypatch, response)
    payload = client.chat_completion(
        messages=[{"role": "user", "content": "Test"}],
        tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
    )

    assert payload["mode"] == "tool"
    assert payload["json"] == tool_arguments
    assert payload["tool_json"] == tool_arguments
    assert payload["raw_tool_arguments"] == json.dumps(tool_arguments, ensure_ascii=False)

    pack_payload = {"note_text": "Example note"}
    result_path = _ingest(
        tmp_path,
        sid="SID789",
        account_id="acct-3",
        pack_payload=pack_payload,
        response_payload=payload,
        monkeypatch=monkeypatch,
    )

    assert result_path.exists()
    stored = json.loads(result_path.read_text(encoding="utf-8").splitlines()[-1])
    stored_analysis = stored["analysis"]
    assert stored_analysis["tone"] == analysis["tone"]
    assert stored_analysis["context_hints"]["topic"] == analysis["context_hints"]["topic"]
    assert stored_analysis["context_hints"]["entities"]["creditor"] == analysis["context_hints"]["entities"]["creditor"]
    assert stored_analysis["risk_flags"] == analysis["risk_flags"]
    assert 0.0 <= stored_analysis["confidence"] <= 0.5
    assert stored["note_metrics"] == {"char_len": 12, "word_len": 2}


def test_try_fix_to_json_handles_none() -> None:
    with pytest.raises(ValueError):
        try_fix_to_json(None)


def test_contract_breach_returns_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="not-json", tool_calls=None)
            )
        ],
        usage=None,
    )

    client, _ = _make_client(monkeypatch, response)
    with pytest.raises(AIClientProtocolError):
        client.chat_completion(messages=[{"role": "user", "content": "Test"}])
