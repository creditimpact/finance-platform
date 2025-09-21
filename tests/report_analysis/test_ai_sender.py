import json
from pathlib import Path

import httpx

from backend.core.logic.report_analysis import ai_sender


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _sample_pack() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
    }


def _config() -> ai_sender.AISenderConfig:
    return ai_sender.AISenderConfig(
        base_url="https://example.test/v1",
        api_key="key-123",
        model="gpt-test",
        timeout=12.0,
    )


def test_send_single_attempt_success() -> None:
    captured: dict[str, object] = {}

    def _request(url, payload, headers, timeout):
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"decision": "merge", "reason": "strong match"}
                            )
                        }
                    }
                ]
            }
        )

    decision, reason = ai_sender.send_single_attempt(
        _sample_pack(),
        _config(),
        request=_request,
    )

    assert decision == "merge"
    assert reason == "strong match"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"] == {
        "Authorization": "Bearer key-123",
        "Content-Type": "application/json",
    }
    assert captured["payload"]["model"] == "gpt-test"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["timeout"] == 12.0


def test_process_pack_retries_then_success(monkeypatch) -> None:
    attempts = {"count": 0}
    delays: list[float] = []
    events: list[tuple[str, dict]] = []

    def _request(url, payload, headers, timeout):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx.HTTPError("temporary failure")
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"decision": "same_debt", "reason": "oc vs ca"}
                            )
                        }
                    }
                ]
            }
        )

    outcome = ai_sender.process_pack(
        _sample_pack(),
        _config(),
        request=_request,
        sleep=delays.append,
        log=lambda event, payload: events.append((event, dict(payload))),
    )

    assert outcome.success is True
    assert outcome.decision == "same_debt"
    assert outcome.reason == "oc vs ca"
    assert outcome.attempts == 3
    assert delays == [1.0, 3.0]
    assert any(event == "SUCCESS" for event, _ in events)


def test_process_pack_failure_records_error() -> None:
    delays: list[float] = []

    def _request(url, payload, headers, timeout):
        raise httpx.ReadTimeout("timed out")

    outcome = ai_sender.process_pack(
        _sample_pack(),
        _config(),
        request=_request,
        sleep=delays.append,
        log=lambda event, payload: None,
    )

    assert outcome.success is False
    assert outcome.error_kind == "ReadTimeout"
    assert outcome.attempts == 4
    assert delays == [1.0, 3.0, 7.0]


def test_write_decision_tags_same_debt(tmp_path: Path) -> None:
    ai_sender.write_decision_tags(
        tmp_path,
        "case-001",
        11,
        16,
        "same_debt",
        "matching oc and ca",
        "2024-06-01T00:00:00Z",
    )

    base = tmp_path / "case-001" / "cases" / "accounts"
    tags_a = json.loads((base / "11" / "tags.json").read_text(encoding="utf-8"))
    tags_b = json.loads((base / "16" / "tags.json").read_text(encoding="utf-8"))

    assert tags_a == [
        {
            "at": "2024-06-01T00:00:00Z",
            "decision": "same_debt",
            "kind": "ai_decision",
            "reason": "matching oc and ca",
            "source": "ai_adjudicator",
            "tag": "ai_decision",
            "with": 16,
        },
        {
            "at": "2024-06-01T00:00:00Z",
            "kind": "same_debt_pair",
            "source": "ai_adjudicator",
            "with": 16,
        },
    ]
    assert tags_b == [
        {
            "at": "2024-06-01T00:00:00Z",
            "decision": "same_debt",
            "kind": "ai_decision",
            "reason": "matching oc and ca",
            "source": "ai_adjudicator",
            "tag": "ai_decision",
            "with": 11,
        },
        {
            "at": "2024-06-01T00:00:00Z",
            "kind": "same_debt_pair",
            "source": "ai_adjudicator",
            "with": 11,
        },
    ]


def test_write_error_tags(tmp_path: Path) -> None:
    ai_sender.write_error_tags(
        tmp_path,
        "case-002",
        21,
        22,
        "Timeout",
        "deadline exceeded",
        "2024-06-01T01:00:00Z",
    )

    base = tmp_path / "case-002" / "cases" / "accounts"
    tags_a = json.loads((base / "21" / "tags.json").read_text(encoding="utf-8"))
    tags_b = json.loads((base / "22" / "tags.json").read_text(encoding="utf-8"))

    expected = {
        "at": "2024-06-01T01:00:00Z",
        "error_kind": "Timeout",
        "kind": "ai_error",
        "message": "deadline exceeded",
        "source": "ai_adjudicator",
    }

    expected_a = dict(expected)
    expected_a["with"] = 22
    expected_b = dict(expected)
    expected_b["with"] = 21

    assert tags_a == [expected_a]
    assert tags_b == [expected_b]

