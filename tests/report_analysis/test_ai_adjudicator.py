from __future__ import annotations

import json as jsonlib

import backend.config as config
from backend.core.logic.report_analysis import ai_adjudicator


def _sample_pack() -> dict:
    return {
        "sid": "case-123",
        "pair": {"a": 11, "b": 16},
        "highlights": {"total": 82, "triggers": ["strong:acctnum"]},
        "context": {"a": ["Creditor A", "Account # 1234"], "b": ["Creditor B", "Account # 5678"]},
        "ids": {"account_number_a": "1234", "account_number_b": "5678"},
        "limits": {"max_lines_per_side": 3},
    }


def _enable_ai(monkeypatch) -> None:
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", True)
    monkeypatch.setattr(config, "AI_TEMPERATURE_DEFAULT", 0.0)
    monkeypatch.setattr(config, "AI_MAX_TOKENS", 256)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("AI_MODEL", "gpt-test")
    monkeypatch.setenv("AI_REQUEST_TIMEOUT", "3")


def test_adjudicate_pair_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENABLE_AI_ADJUDICATOR", False)

    called = False

    def _fake_post(*args, **kwargs):  # pragma: no cover - sanity guard
        nonlocal called
        called = True
        raise AssertionError("HTTP should not be called when disabled")

    monkeypatch.setattr(ai_adjudicator.httpx, "post", _fake_post)

    base = tmp_path / "case-123" / "cases" / "accounts"
    legacy_a = base / "11" / "ai" / "pack_pair_11_16.json"
    legacy_b = base / "16" / "ai" / "pack_pair_16_11.json"
    legacy_a.parent.mkdir(parents=True, exist_ok=True)
    legacy_b.parent.mkdir(parents=True, exist_ok=True)
    legacy_a.write_text("legacy", encoding="utf-8")
    legacy_b.write_text("legacy", encoding="utf-8")

    pack = _sample_pack()
    resp = ai_adjudicator.adjudicate_pair(pack)

    assert resp == {
        "decision": "ai_disabled",
        "confidence": 0.0,
        "reason": "AI adjudication disabled",
        "reasons": ["AI adjudication disabled"],
        "flags": {"account_match": "unknown", "debt_match": "unknown"},
    }
    assert not called

    ai_adjudicator.persist_ai_decision("case-123", tmp_path, 11, 16, resp)

    tags_a_path = base / "11" / "tags.json"
    tags_b_path = base / "16" / "tags.json"

    tags_a = jsonlib.loads(tags_a_path.read_text(encoding="utf-8"))
    tags_b = jsonlib.loads(tags_b_path.read_text(encoding="utf-8"))

    expected_tag_a = {
        "kind": "merge_result",
        "with": 16,
        "decision": "ai_disabled",
        "confidence": 0.0,
        "reason": "AI adjudication disabled",
        "reasons": ["AI adjudication disabled"],
        "flags": {"account_match": "unknown", "debt_match": "unknown"},
        "source": "ai_adjudicator",
    }
    expected_tag_b = dict(expected_tag_a)
    expected_tag_b["with"] = 11

    assert tags_a == [expected_tag_a]
    assert tags_b == [expected_tag_b]
    assert not legacy_a.exists()
    assert not legacy_b.exists()
    assert not (base / "11" / "ai").exists()
    assert not (base / "16" / "ai").exists()


def test_adjudicate_pair_enabled_and_persist(monkeypatch, tmp_path):
    _enable_ai(monkeypatch)

    captured: dict = {}

    def _fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": jsonlib.dumps(
                                    {
                                        "decision": "merge",
                                        "confidence": 0.83,
                                        "reason": "accounts align",
                                        "reasons": ["matched creditor names"],
                                        "flags": {
                                            "account_match": True,
                                            "debt_match": True,
                                        },
                                    }
                                )
                            }
                        }
                    ]
                }

        return _Resp()

    monkeypatch.setattr(ai_adjudicator.httpx, "post", _fake_post)

    base = tmp_path / "case-123" / "cases" / "accounts"
    legacy_a = base / "11" / "ai" / "pack_pair_11_16.json"
    legacy_b = base / "16" / "ai" / "pack_pair_16_11.json"
    legacy_a.parent.mkdir(parents=True, exist_ok=True)
    legacy_b.parent.mkdir(parents=True, exist_ok=True)
    legacy_a.write_text("legacy", encoding="utf-8")
    legacy_b.write_text("legacy", encoding="utf-8")

    pack = _sample_pack()
    resp = ai_adjudicator.adjudicate_pair(pack)

    assert resp == {
        "decision": "merge",
        "confidence": 0.83,
        "reason": "accounts align",
        "reasons": ["matched creditor names"],
        "flags": {"account_match": True, "debt_match": True},
    }

    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["payload"]["model"] == "gpt-test"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["timeout"] == 3.0

    ai_adjudicator.persist_ai_decision("case-123", tmp_path, 11, 16, resp)

    tags_a_path = base / "11" / "tags.json"
    tags_b_path = base / "16" / "tags.json"

    tags_a = jsonlib.loads(tags_a_path.read_text(encoding="utf-8"))
    tags_b = jsonlib.loads(tags_b_path.read_text(encoding="utf-8"))

    expected_a = {
        "kind": "merge_result",
        "with": 16,
        "decision": "merge",
        "confidence": 0.83,
        "reason": "accounts align",
        "reasons": ["matched creditor names"],
        "flags": {"account_match": True, "debt_match": True},
        "source": "ai_adjudicator",
    }
    expected_b = {
        "kind": "merge_result",
        "with": 11,
        "decision": "merge",
        "confidence": 0.83,
        "reason": "accounts align",
        "reasons": ["matched creditor names"],
        "flags": {"account_match": True, "debt_match": True},
        "source": "ai_adjudicator",
    }

    assert tags_a == [expected_a]
    assert tags_b == [expected_b]
    assert not legacy_a.exists()
    assert not legacy_b.exists()
    assert not (base / "11" / "ai").exists()
    assert not (base / "16" / "ai").exists()


def test_adjudicate_pair_enabled_no_merge(monkeypatch, tmp_path):
    _enable_ai(monkeypatch)

    def _fake_post(url, json=None, headers=None, timeout=None):
        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": jsonlib.dumps(
                                    {
                                        "decision": "different",
                                        "confidence": 0.41,
                                        "reason": "conflicting payment history",
                                        "flags": {
                                            "account_match": False,
                                            "debt_match": False,
                                        },
                                    }
                                )
                            }
                        }
                    ]
                }

        return _Resp()

    monkeypatch.setattr(ai_adjudicator.httpx, "post", _fake_post)

    pack = _sample_pack()
    resp = ai_adjudicator.adjudicate_pair(pack)

    assert resp == {
        "decision": "different",
        "confidence": 0.41,
        "reason": "conflicting payment history",
        "reasons": ["conflicting payment history"],
        "flags": {"account_match": False, "debt_match": False},
    }

    ai_adjudicator.persist_ai_decision("case-123", tmp_path, 11, 16, resp)

    base = tmp_path / "case-123" / "cases" / "accounts"
    tags_a_path = base / "11" / "tags.json"
    tags_b_path = base / "16" / "tags.json"

    tags_a = jsonlib.loads(tags_a_path.read_text(encoding="utf-8"))
    tags_b = jsonlib.loads(tags_b_path.read_text(encoding="utf-8"))

    expected_a = {
        "kind": "merge_result",
        "with": 16,
        "decision": "different",
        "confidence": 0.41,
        "reason": "conflicting payment history",
        "reasons": ["conflicting payment history"],
        "flags": {"account_match": False, "debt_match": False},
        "source": "ai_adjudicator",
    }
    expected_b = {
        "kind": "merge_result",
        "with": 11,
        "decision": "different",
        "confidence": 0.41,
        "reason": "conflicting payment history",
        "reasons": ["conflicting payment history"],
        "flags": {"account_match": False, "debt_match": False},
        "source": "ai_adjudicator",
    }

    assert tags_a == [expected_a]
    assert tags_b == [expected_b]
    assert not (base / "11" / "ai").exists()
    assert not (base / "16" / "ai").exists()
