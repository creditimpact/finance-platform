from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
import unicodedata

from backend.ai.note_style_stage import build_note_style_pack_for_account
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths


_ZERO_WIDTH_TRANSLATION = {
    ord("\u200b"): " ",
    ord("\u200c"): " ",
    ord("\u200d"): " ",
    ord("\ufeff"): " ",
    ord("\u2060"): " ",
}


def _sanitize_note_text(note: str) -> str:
    normalized = unicodedata.normalize("NFKC", note)
    translated = normalized.translate(_ZERO_WIDTH_TRANSLATION)
    return " ".join(translated.split()).strip()


def _normalized_hash(text: str) -> str:
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _expected_salt(sid: str, account_id: str, analysis: dict[str, object]) -> str:
    tone_value = str(analysis.get("tone") or "").strip().lower()
    if not tone_value:
        tone_value = "neutral"

    context = analysis.get("context_hints")
    if not isinstance(context, dict):
        context = {}

    topic_value = str(context.get("topic") or "").strip().lower()
    if not topic_value:
        topic_value = "other"

    timeframe = context.get("timeframe") if isinstance(context, dict) else None
    if not isinstance(timeframe, dict):
        timeframe = {}

    relative = str(timeframe.get("relative") or "").strip().lower()
    if relative:
        timeframe_bucket = f"relative:{relative}"
    else:
        month_text = str(timeframe.get("month") or "").strip()
        timeframe_bucket = f"month:{month_text}" if month_text else "none"

    entities = context.get("entities") if isinstance(context, dict) else None
    if not isinstance(entities, dict):
        entities = {}

    amount_band = "none"
    amount_value = entities.get("amount")
    try:
        parsed_amount = float(amount_value)
    except (TypeError, ValueError):
        parsed_amount = None

    if parsed_amount is not None and math.isfinite(parsed_amount):
        absolute = abs(parsed_amount)
        if absolute == 0:
            amount_band = "zero"
        elif absolute < 100:
            amount_band = "lt_100"
        elif absolute < 500:
            amount_band = "100_499"
        elif absolute < 1000:
            amount_band = "500_999"
        elif absolute < 5000:
            amount_band = "1000_4999"
        elif absolute < 10000:
            amount_band = "5000_9999"
        else:
            amount_band = "gte_10000"

    emphasis_values = analysis.get("emphasis")
    emphasis_sorted: list[str] = []
    if isinstance(emphasis_values, list):
        seen: set[str] = set()
        for entry in emphasis_values:
            text = str(entry or "").strip()
            if not text:
                continue
            seen.add(text)
        emphasis_sorted = sorted(seen)

    payload = {
        "sid": sid,
        "account_id": account_id,
        "tone": tone_value,
        "topic": topic_value,
        "buckets": {
            "timeframe_bucket": timeframe_bucket,
            "amount_band": amount_band,
            "emphasis_sorted": emphasis_sorted,
        },
    }

    message = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hmac.new(b"tests-note-style-pepper", message, hashlib.sha256).hexdigest()[:12]


def _write_response(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_note_style_stage_builds_artifacts(tmp_path: Path) -> None:
    sid = "SID001"
    account_id = "idx-001"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "Please help, the bank made an error and I already paid this account."

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": note},
            "received_at": "2024-01-01T00:00:00Z",
        },
    )

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert result["status"] == "completed"

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    assert account_paths.pack_file.is_file()
    assert account_paths.result_file.is_file()
    pack_payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8"))
    result_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))

    sanitized = _sanitize_note_text(note)
    expected_hash = _normalized_hash(sanitized)
    expected_short_hash = expected_hash[:12]

    pack_messages = pack_payload["messages"]
    assert isinstance(pack_messages, list)
    assert pack_messages[0]["role"] == "system"
    assert "tone" in pack_messages[0]["content"].lower()
    assert pack_messages[1]["role"] == "user"
    user_content = pack_messages[1]["content"]
    assert sanitized not in user_content
    user_payload = json.loads(user_content)
    extractor_payload = user_payload["extractor"]
    assert extractor_payload["tone"] == "empathetic"
    assert extractor_payload["context_hints"]["topic"] == "payment_dispute"
    assert set(["paid_already", "inaccurate_reporting"]).issubset(
        set(extractor_payload["emphasis"])
    )
    assert extractor_payload["confidence"] >= 0.5

    assert "prompt_salt" not in pack_payload
    analysis_payload = result_payload["analysis"]
    expected_salt = _expected_salt(sid, account_id, analysis_payload)
    assert user_payload["prompt_salt"] == expected_salt
    assert expected_salt in pack_messages[0]["content"]
    assert result_payload["prompt_salt"] == expected_salt
    assert len(expected_salt) == 12
    assert pack_payload["note_hash"] == expected_short_hash
    assert pack_payload["model"] == "gpt-4o-mini"
    assert (
        pack_payload["source_response_path"]
        == f"runs/{sid}/frontend/review/responses/{account_id}.result.json"
    )
    assert sanitized not in account_paths.result_file.read_text(encoding="utf-8")
    assert result_payload["note_hash"] == expected_short_hash
    assert result_payload["source_hash"] == expected_hash
    assert result_payload["analysis"] == result_payload["extractor"]
    analysis_payload = result_payload["analysis"]
    assert pack_payload["extractor"] == analysis_payload
    assert analysis_payload["tone"] == "empathetic"
    assert analysis_payload["context_hints"]["topic"] == "payment_dispute"
    assert analysis_payload["context_hints"]["entities"]["creditor"] is None
    assert set(["paid_already", "inaccurate_reporting", "support_request"]).issubset(
        set(analysis_payload["emphasis"])
    )
    assert analysis_payload["confidence"] >= 0.5
    assert analysis_payload["risk_flags"] == []

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    assert index_payload["schema_version"] == 1
    assert index_payload["root"] == "."
    assert index_payload["packs_dir"] == "packs"
    assert index_payload["results_dir"] == "results"
    packs = index_payload["packs"]
    assert len(packs) == 1
    first_entry = packs[0]
    assert first_entry["account_id"] == account_id
    assert first_entry["status"] == "built"
    expected_pack = account_paths.pack_file.relative_to(paths.base).as_posix()
    assert first_entry["pack"] == expected_pack
    assert first_entry["source_hash"] == expected_short_hash
    assert first_entry["lines"] == 1
    assert first_entry["built_at"]

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "success"
    assert note_style_stage["summary"]["packs_completed"] == 1


def test_note_style_stage_idempotent_for_unchanged_response(tmp_path: Path) -> None:
    sid = "SID002"
    account_id = "idx-002"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "Please help fix this error."

    response_path = response_dir / f"{account_id}.result.json"
    _write_response(
        response_path,
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": note},
            "received_at": "2024-01-02T00:00:00Z",
        },
    )

    first = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert first["status"] == "completed"

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    initial_pack = account_paths.pack_file.read_text(encoding="utf-8")
    initial_result = account_paths.result_file.read_text(encoding="utf-8")
    initial_index = paths.index_file.read_text(encoding="utf-8")

    second = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert second["status"] == "unchanged"
    assert account_paths.pack_file.read_text(encoding="utf-8") == initial_pack
    assert account_paths.result_file.read_text(encoding="utf-8") == initial_result
    assert paths.index_file.read_text(encoding="utf-8") == initial_index


def test_note_style_stage_updates_on_modified_note(tmp_path: Path) -> None:
    sid = "SID003"
    account_id = "idx-003"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    response_path = response_dir / f"{account_id}.result.json"
    _write_response(
        response_path,
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please help correct this."},
            "received_at": "2024-01-03T00:00:00Z",
        },
    )

    first = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    first_salt = first["prompt_salt"]

    _write_response(
        response_path,
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "This is urgent and I dispute this account."},
            "received_at": "2024-01-03T00:10:00Z",
        },
    )

    updated = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert updated["status"] == "completed"
    assert updated["prompt_salt"] != first_salt

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    updated_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    assert updated_payload["analysis"]["tone"] == "assertive"
    assert updated_payload["analysis"]["context_hints"]["topic"] == "payment_dispute"

def test_note_style_stage_sanitizes_note_text(tmp_path: Path) -> None:
    sid = "SID004"
    account_id = "idx-004"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "  Héllo\u00a0Bank\u2009\nAlready\u00a0paid  "

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": note},
            "received_at": "2024-01-04T00:00:00Z",
        },
    )

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert result["status"] == "completed"

    sanitized = _sanitize_note_text(note)
    expected_hash = _normalized_hash(sanitized)
    expected_short_hash = expected_hash[:12]

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8"))
    result_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    pack_user_payload = json.loads(pack_payload["messages"][1]["content"])
    assert "prompt_salt" not in pack_payload
    analysis_payload = result_payload["analysis"]
    expected_salt = _expected_salt(sid, account_id, analysis_payload)
    assert pack_user_payload["prompt_salt"] == expected_salt
    extractor_payload = pack_user_payload["extractor"]
    assert extractor_payload["context_hints"]["topic"] == "payment_dispute"
    assert extractor_payload["emphasis"] == ["paid_already"]
    assert extractor_payload["confidence"] >= 0.5
    assert sanitized not in pack_payload["messages"][1]["content"]
    assert pack_payload["note_hash"] == expected_short_hash
    assert result_payload["source_hash"] == expected_hash
    assert result_payload["note_hash"] == expected_short_hash
    assert result_payload["prompt_salt"] == expected_salt
    assert result_payload["analysis"] == result_payload["extractor"]
    assert result_payload["analysis"]["context_hints"]["topic"] == "payment_dispute"
    assert result_payload["analysis"]["emphasis"] == ["paid_already"]
    assert result_payload["analysis"]["confidence"] >= 0.5
    assert result["prompt_salt"] == expected_salt


def test_note_style_stage_skips_when_note_sanitizes_empty(tmp_path: Path) -> None:
    sid = "SID005"
    account_id = "idx-005"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    note = "\u200b\n\t  "

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": note},
            "received_at": "2024-01-05T00:00:00Z",
        },
    )

    result = build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    assert result["status"] == "skipped"
    assert result["reason"] == "empty_note"

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    assert not account_paths.pack_file.exists()
    assert not account_paths.result_file.exists()

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    assert index_payload["packs"] == []
    assert index_payload["root"] == "."

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "success"
    assert note_style_stage["summary"]["packs_completed"] == 0
