from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import pytest

from backend.ai.note_style_sender import send_note_style_packs_for_sid
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths
from ._helpers import prime_stage, stage_view


def _analysis_payload(*, tone: str, topic: str, emphasis: Iterable[str], confidence: float) -> dict[str, object]:
    return {
        "tone": tone,
        "context_hints": {
            "timeframe": {"month": "May", "relative": "recent"},
            "topic": topic,
            "entities": {"creditor": "Example Bank", "amount": 310.75},
        },
        "emphasis": list(emphasis),
        "confidence": confidence,
        "risk_flags": ["follow_up"],
    }


class _SequencedClient:
    def __init__(self, responses: Mapping[str, Iterable[str]]) -> None:
        self._responses: dict[str, list[str]] = {
            account: list(sequence) for account, sequence in responses.items()
        }
        self._counters: defaultdict[str, int] = defaultdict(int)
        self.calls: list[Mapping[str, object]] = []

    def chat_completion(self, *, model, messages, temperature, **kwargs):  # type: ignore[override]
        account_id = self._infer_account_id(messages)
        if account_id is None:
            for candidate, sequence in self._responses.items():
                if self._counters[candidate] < len(sequence):
                    account_id = candidate
                    break
        if account_id is None:
            raise AssertionError("Unable to determine account for chat completion call")

        index = self._counters[account_id]
        self._counters[account_id] += 1
        payloads = self._responses.get(account_id)
        if not payloads:
            raise AssertionError(f"No response configured for account {account_id}")
        if index >= len(payloads):
            content = payloads[-1]
        else:
            content = payloads[index]

        self.calls.append({
            "account_id": account_id,
            "model": model,
            "messages": messages,
            "kwargs": kwargs,
        })
        return {"choices": [{"message": {"content": content}}]}

    @staticmethod
    def _infer_account_id(messages: Iterable[Mapping[str, object]]) -> str | None:
        for entry in messages:
            content = entry.get("content")
            if isinstance(content, str) and "payload:" in content:
                return content.split("payload:", 1)[1].strip()
        return None


def _read_single_jsonl(path: Path) -> dict[str, object]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1, f"expected single jsonl line in {path}"
    return json.loads(lines[0])


def test_note_style_integration_parses_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID-INTEGRATION"
    accounts = [
        "idx-501",  # clean JSON
        "idx-502",  # fenced JSON
        "idx-503",  # truncated then success
        "idx-504",  # prose failure
    ]

    prime_stage(
        tmp_path,
        sid,
        expected_accounts=accounts,
        built_accounts=accounts,
    )

    clean_analysis = _analysis_payload(
        tone="Warm",
        topic="Plan discussion",
        emphasis=["gratitude"],
        confidence=0.81,
    )
    fenced_analysis = _analysis_payload(
        tone="Calm",
        topic="Payment timeline",
        emphasis=["timeline"],
        confidence=0.76,
    )
    retry_analysis = _analysis_payload(
        tone="Assured",
        topic="Confirmation",
        emphasis=["confirmation"],
        confidence=0.63,
    )

    truncated_attempt = json.dumps(retry_analysis)[:-2]
    prose_attempt = "Sure, here's a quick summary without any JSON structure."

    client = _SequencedClient(
        {
            accounts[0]: [json.dumps(clean_analysis)],
            accounts[1]: [
                "Here is the result you requested:\n```json\n"
                + json.dumps({"analysis": fenced_analysis})
                + "\n```\nThanks!"
            ],
            accounts[2]: [truncated_attempt, json.dumps(retry_analysis)],
            accounts[3]: [prose_attempt, prose_attempt, prose_attempt],
        }
    )

    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)
    monkeypatch.setattr(
        "backend.ai.note_style_sender.config.NOTE_STYLE_INVALID_RESULT_RETRY_ATTEMPTS",
        2,
    )
    monkeypatch.setattr(
        "backend.ai.note_style_sender.config.NOTE_STYLE_INVALID_RESULT_RETRY_TOOL_CALL",
        False,
    )
    monkeypatch.setattr("backend.ai.note_style_sender.config.NOTE_STYLE_SKIP_IF_RESULT_EXISTS", False)
    monkeypatch.setattr(
        "backend.ai.note_style_sender.config.NOTE_STYLE_IDEMPOTENT_BY_NOTE_HASH", False
    )

    processed = send_note_style_packs_for_sid(sid, runs_root=tmp_path)

    assert set(processed) == {accounts[0], accounts[1], accounts[2]}
    assert len(client.calls) == 7

    paths = ensure_note_style_paths(tmp_path, sid, create=False)

    clean_paths = ensure_note_style_account_paths(paths, accounts[0], create=False)
    clean_result = _read_single_jsonl(clean_paths.result_file)
    assert clean_result["sid"] == sid
    assert clean_result["account_id"] == accounts[0]
    assert clean_result["analysis"]["tone"] == "Warm"
    assert not clean_paths.result_raw_file.exists()

    fenced_paths = ensure_note_style_account_paths(paths, accounts[1], create=False)
    fenced_result = _read_single_jsonl(fenced_paths.result_file)
    assert fenced_result["analysis"]["tone"] == "Calm"
    assert not fenced_paths.result_raw_file.exists()

    retry_paths = ensure_note_style_account_paths(paths, accounts[2], create=False)
    retry_result = _read_single_jsonl(retry_paths.result_file)
    assert retry_result["analysis"]["tone"] == "Assured"
    retry_raw_text = retry_paths.result_raw_file.read_text(encoding="utf-8")
    assert truncated_attempt in retry_raw_text

    failure_paths = ensure_note_style_account_paths(paths, accounts[3], create=False)
    failure_result = _read_single_jsonl(failure_paths.result_file)
    assert failure_result["error"] == "invalid_result"
    assert failure_result.get("status") == "failed"
    assert failure_result.get("parser_reason")
    failure_raw_text = failure_paths.result_raw_file.read_text(encoding="utf-8")
    assert prose_attempt in failure_raw_text

    view = stage_view(tmp_path, sid)
    assert set(view.packs_expected) == set(accounts)
    assert set(view.packs_built) == set(accounts)
    assert set(view.packs_completed) == {accounts[0], accounts[1], accounts[2]}
    assert set(view.packs_failed) == {accounts[3]}
    assert view.state == "success"
    assert view.built_complete is True
