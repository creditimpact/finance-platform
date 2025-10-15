from __future__ import annotations

import json

from backend.frontend.packs.responses import append_frontend_response


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_append_frontend_response_default_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("FRONTEND_PACKS_RESPONSES_DIR", raising=False)
    monkeypatch.delenv("FRONTEND_PACKS_STAGE_DIR", raising=False)
    monkeypatch.delenv("FRONTEND_RESPONSES_DIR", raising=False)

    run_dir = tmp_path / "runs" / "SID-123"
    payload = {"answer": "yes"}

    append_frontend_response(run_dir, "acct-001", payload)

    stage_file = run_dir / "frontend" / "review" / "responses" / "acct-001.jsonl"
    legacy_file = run_dir / "frontend" / "responses" / "acct-001.jsonl"

    assert stage_file.exists()
    assert legacy_file.exists()
    assert _read_jsonl(stage_file) == [payload]
    assert _read_jsonl(legacy_file) == [payload]


def test_append_frontend_response_sanitizes_account_id(tmp_path, monkeypatch):
    monkeypatch.delenv("FRONTEND_PACKS_RESPONSES_DIR", raising=False)
    monkeypatch.delenv("FRONTEND_PACKS_STAGE_DIR", raising=False)
    monkeypatch.delenv("FRONTEND_RESPONSES_DIR", raising=False)

    run_dir = tmp_path / "runs" / "SID-456"

    append_frontend_response(run_dir, " acct/Needs Sanitize ", {"ok": True})

    stage_file = run_dir / "frontend" / "review" / "responses" / "acct_Needs_Sanitize.jsonl"
    assert stage_file.exists()


def test_append_frontend_response_custom_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTEND_PACKS_RESPONSES_DIR", "custom/responses")
    monkeypatch.delenv("FRONTEND_RESPONSES_DIR", raising=False)

    run_dir = tmp_path / "runs" / "SID-env"
    payload = {"value": 1}

    append_frontend_response(run_dir, "acct-xyz", payload)

    stage_file = run_dir / "custom" / "responses" / "acct-xyz.jsonl"
    legacy_file = run_dir / "frontend" / "responses" / "acct-xyz.jsonl"

    assert stage_file.exists()
    assert _read_jsonl(stage_file) == [payload]
    assert legacy_file.exists()


def test_append_frontend_response_same_stage_and_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTEND_PACKS_RESPONSES_DIR", "frontend/responses")
    monkeypatch.delenv("FRONTEND_RESPONSES_DIR", raising=False)

    run_dir = tmp_path / "runs" / "SID-same"
    payload = {"value": "first"}

    append_frontend_response(run_dir, "acct", payload)

    target_file = run_dir / "frontend" / "responses" / "acct.jsonl"
    assert target_file.exists()
    assert _read_jsonl(target_file) == [payload]
