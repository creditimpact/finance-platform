import json
import logging
from collections.abc import Mapping
from pathlib import Path

import pytest

from backend.core.ai.paths import get_merge_paths, pair_pack_filename, pair_result_filename
from backend.core.logic.report_analysis.tags_compact import compact_tags_for_account
from backend.pipeline.runs import RunManifest
from backend.pipeline.runs import RUNS_ROOT_ENV
from scripts import send_ai_merge_packs
from scripts.build_ai_merge_packs import main as build_ai_merge_packs_main


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_raw_lines(path: Path, lines: list[str]) -> None:
    _write_json(path, [{"text": line} for line in lines])


def _assert_pack_messages_with_rules(
    pack: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    actual_messages = pack.get("messages")
    expected_messages = expected.get("messages")
    assert isinstance(actual_messages, list)
    assert isinstance(expected_messages, list)
    assert len(actual_messages) == len(expected_messages)
    assert actual_messages[1:] == expected_messages[1:]
    actual_system = actual_messages[0]
    expected_system = expected_messages[0]
    assert isinstance(actual_system, Mapping)
    assert isinstance(expected_system, Mapping)
    assert actual_system.get("role") == expected_system.get("role")
    expected_content = expected_system.get("content")
    actual_content = actual_system.get("content")
    assert isinstance(expected_content, str)
    assert isinstance(actual_content, str)
    trimmed_expected = expected_content.rstrip()
    assert actual_content.startswith(trimmed_expected)
    assert "Debt rules:" in actual_content


def _create_merge_pack(
    runs_root: Path,
    sid: str,
    pack_payload: Mapping[str, object],
    *,
    a_idx: int = 11,
    b_idx: int = 16,
    index_entry: Mapping[str, object] | None = None,
) -> tuple[dict[str, Path], Path]:
    merge_paths = get_merge_paths(runs_root, sid, create=True)
    packs_dir = merge_paths.packs_dir
    pack_filename = pair_pack_filename(a_idx, b_idx)
    pack_path = packs_dir / pack_filename
    pack_path.write_text(
        json.dumps(pack_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    entry = {
        "a": a_idx,
        "b": b_idx,
        "pair": [a_idx, b_idx],
        "pack_file": pack_filename,
        "lines_a": 0,
        "lines_b": 0,
        "score_total": 0,
    }
    if index_entry:
        entry.update(dict(index_entry))

    index_payload = {
        "sid": sid,
        "packs": [dict(entry)],
        "pairs": [dict(entry)],
        "pairs_count": 1,
    }
    merge_paths.index_file.write_text(
        json.dumps(index_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    return merge_paths, pack_path


def _normalize_flag_values(raw_flags: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in raw_flags.items():
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "false"}:
                normalized[key] = lowered == "true"
            else:
                normalized[key] = value
        else:
            normalized[key] = value
    return normalized


def _normalize_tags(tags: list[Mapping[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for tag in tags:
        normalized_tag = dict(tag)
        flags = tag.get("flags")
        if isinstance(flags, Mapping):
            normalized_tag["flags"] = _normalize_flag_values(flags)
        normalized.append(normalized_tag)
    return normalized


def _merge_pair_tag(partner: int) -> dict:
    return {
        "tag": "merge_pair",
        "kind": "merge_pair",
        "source": "merge_scorer",
        "with": partner,
        "decision": "ai",
        "total": 59,
        "mid": 20,
        "dates_all": False,
        "parts": {"balance_owed": 31, "account_number": 28},
        "aux": {
            "acctnum_level": "exact_or_known_match",
            "matched_fields": {
                "balance_owed": True,
                "last_payment": True,
                "account_number": True,
            },
            "by_field_pairs": {"account_number": ["transunion", "experian"]},
            "acctnum_digits_len_a": 12,
            "acctnum_digits_len_b": 12,
        },
        "conflicts": ["credit_limit:conflict"],
        "strong": True,
        "matched_pairs": {"account_number": ["transunion", "experian"]},
        "acctnum_digits_len_a": 12,
        "acctnum_digits_len_b": 12,
    }


def _merge_best_tag(partner: int) -> dict:
    return {
        "tag": "merge_best",
        "kind": "merge_best",
        "source": "merge_scorer",
        "with": partner,
        "decision": "ai",
        "total": 59,
        "mid": 20,
        "parts": {"balance_owed": 31, "account_number": 28},
        "aux": {
            "acctnum_level": "exact_or_known_match",
            "matched_fields": {
                "balance_owed": True,
                "last_payment": True,
                "account_number": True,
            },
            "by_field_pairs": {"account_number": ["transunion", "experian"]},
            "acctnum_digits_len_a": 12,
            "acctnum_digits_len_b": 12,
        },
        "conflicts": ["credit_limit:conflict"],
        "strong": True,
        "matched_pairs": {"account_number": ["transunion", "experian"]},
        "acctnum_digits_len_a": 12,
        "acctnum_digits_len_b": 12,
    }


@pytest.fixture()
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.mark.parametrize(
    ("dates_plausible", "expected_decision", "expected_account_flag"),
    [
        (True, "same_debt_diff_account", False),
        (False, "same_debt_account_unknown", "unknown"),
    ],
)
def test_send_ai_merge_packs_forces_ca_pairs_to_same_debt(
    monkeypatch: pytest.MonkeyPatch,
    runs_root: Path,
    dates_plausible: bool,
    expected_decision: str,
    expected_account_flag: object,
) -> None:
    sid = f"ca-equal-{int(dates_plausible)}"
    pack_payload = {
        "sid": sid,
        "pair": {"a": 41, "b": 42},
        "context": {"a": [], "b": []},
        "highlights": {"total": 0},
        "context_flags": {
            "is_collection_agency_a": True,
            "is_collection_agency_b": True,
            "amounts_equal_within_tol": True,
            "dates_plausible_chain": dates_plausible,
        },
        "messages": [
            {"role": "system", "content": "Test system"},
            {
                "role": "user",
                "content": json.dumps(
                    {"sid": sid, "pair": {"a": 41, "b": 42}}, ensure_ascii=False
                ),
            },
        ],
    }

    merge_paths, _ = _create_merge_pack(
        runs_root,
        sid,
        pack_payload,
        a_idx=41,
        b_idx=42,
    )

    manifest = RunManifest.for_sid(sid)
    manifest.update_ai_packs(
        dir=merge_paths.base,
        index=merge_paths.index_file,
        logs=merge_paths.log_file,
        pairs=1,
    )

    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    def _fake_decide(pack: dict, *, timeout: float):
        assert pack["pair"] == {"a": 41, "b": 42}
        return {
            "decision": "different",
            "reason": "Identifiers differ",
            "flags": {"account_match": False, "debt_match": False},
        }

    monkeypatch.setattr(send_ai_merge_packs, "decide_merge_or_different", _fake_decide)
    monkeypatch.setattr(
        send_ai_merge_packs,
        "_isoformat_timestamp",
        lambda dt=None: "2024-07-04T00:00:00Z",
    )

    send_ai_merge_packs.main(["--sid", sid, "--runs-root", str(runs_root)])

    result_path = merge_paths.results_dir / pair_result_filename(41, 42)
    assert result_path.exists()
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_payload["decision"] == expected_decision
    assert result_payload["flags"]["debt_match"] is True
    assert result_payload["flags"]["account_match"] == expected_account_flag
    assert "Collection agencies" in result_payload["reason"]

    base_accounts = runs_root / sid / "cases" / "accounts"
    tags_a = _normalize_tags(
        json.loads((base_accounts / "41" / "tags.json").read_text(encoding="utf-8"))
    )
    tags_b = _normalize_tags(
        json.loads((base_accounts / "42" / "tags.json").read_text(encoding="utf-8"))
    )

    decision_tag_a = next(tag for tag in tags_a if tag.get("kind") == "ai_decision")
    decision_tag_b = next(tag for tag in tags_b if tag.get("kind") == "ai_decision")
    assert decision_tag_a["decision"] == expected_decision
    assert decision_tag_b["decision"] == expected_decision
    assert decision_tag_a["flags"]["account_match"] == expected_account_flag
    assert decision_tag_b["flags"]["account_match"] == expected_account_flag

    same_debt_tag = next(tag for tag in tags_a if tag.get("kind") == "same_debt_pair")
    assert same_debt_tag["with"] == 42



def test_send_ai_merge_packs_records_merge_decision(
    monkeypatch: pytest.MonkeyPatch,
    runs_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sid = "codex-smoke"
    accounts_root = runs_root / sid / "cases" / "accounts"
    account_a_dir = accounts_root / "11"
    account_b_dir = accounts_root / "16"

    _write_raw_lines(
        account_a_dir / "raw_lines.json",
        [
            "US BK CACS",
            "Transunion ® Experian ® Equifax ®",
            "Account # 409451****** -- 409451******",
            "Balance Owed: $12,091 -- $12,091",
            "Two-Year Payment History: 111100001111",
            "Creditor Remarks: Late due to pandemic",
        ],
    )
    _write_raw_lines(
        account_b_dir / "raw_lines.json",
        [
            "U S BANK",
            "Account # -- 409451******",
            "Balance Owed: -- $12,091 --",
            "Past Due Amount: --",
            "Last Payment: 13.9.2024",
            "Days Late - 7 Year History: 0000000",
        ],
    )

    _write_json(account_a_dir / "tags.json", [_merge_pair_tag(16), _merge_best_tag(16)])
    _write_json(account_b_dir / "tags.json", [_merge_best_tag(11)])

    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="scripts.build_ai_merge_packs"):
        build_ai_merge_packs_main(
            ["--sid", sid, "--runs-root", str(runs_root), "--max-lines-per-side", "6"]
        )

    build_messages = [record.getMessage() for record in caplog.records]
    assert any(
        f"MANIFEST_AI_PACKS_UPDATED sid={sid}" in message for message in build_messages
    )

    manifest = RunManifest.for_sid(sid)
    packs_info = manifest.data.get("ai", {}).get("packs", {})
    status_info = manifest.data.get("ai", {}).get("status", {})
    merge_paths = get_merge_paths(runs_root, sid, create=False)
    packs_dir = merge_paths.packs_dir.resolve()
    base_dir = merge_paths.base.resolve()
    logs_path = merge_paths.log_file.resolve()
    results_dir = merge_paths.results_dir.resolve()
    assert Path(packs_info.get("dir")) == base_dir
    assert Path(packs_info.get("dir")).exists()
    assert Path(packs_info.get("index")).exists()
    assert packs_info.get("pairs") >= 1
    assert status_info.get("built") is True
    assert status_info.get("sent") is False
    assert status_info.get("compacted") is False
    assert status_info.get("skipped_reason") is None
    
    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    def _fake_decide(pack: dict, *, timeout: float):
        assert pack["pair"] == {"a": 11, "b": 16}
        return {
            "decision": "merge",
            "reason": "Records align cleanly.",
            "flags": {"account_match": True, "debt_match": True},
        }

    monkeypatch.setattr(send_ai_merge_packs, "decide_merge_or_different", _fake_decide)
    monkeypatch.setattr(
        send_ai_merge_packs,
        "_isoformat_timestamp",
        lambda dt=None: "2024-07-01T09:30:00Z",
    )

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="scripts.send_ai_merge_packs"):
        send_ai_merge_packs.main(["--sid", sid, "--runs-root", str(runs_root)])

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        f"SENDER_PACKS_DIR_FROM_MANIFEST sid={sid} dir=" in message
        for message in messages
    )
    assert any(f"MANIFEST_AI_SENT sid={sid}" in message for message in messages)
    assert any(f"MANIFEST_AI_COMPACTED sid={sid}" in message for message in messages)

    manifest = RunManifest.for_sid(sid)
    status_info = manifest.data.get("ai", {}).get("status", {})
    assert status_info.get("sent") is True
    assert status_info.get("compacted") is True
    assert status_info.get("skipped_reason") is None
    packs_info = manifest.data.get("ai", {}).get("packs", {})
    assert Path(packs_info.get("dir")).exists()
    assert Path(packs_info.get("index")).exists()
    assert packs_info.get("pairs") >= 1

    result_path = results_dir / pair_result_filename(11, 16)
    assert result_path.exists()

    account_a_tags = json.loads((account_a_dir / "tags.json").read_text(encoding="utf-8"))
    account_b_tags = json.loads((account_b_dir / "tags.json").read_text(encoding="utf-8"))

    decision_tag_a = next(
        tag
        for tag in account_a_tags
        if tag.get("kind") == "ai_decision" and tag.get("with") == 16
    )
    decision_tag_b = next(
        tag
        for tag in account_b_tags
        if tag.get("kind") == "ai_decision" and tag.get("with") == 11
    )

    expected_decision = {
        "kind": "ai_decision",
        "tag": "ai_decision",
        "source": "ai_adjudicator",
        "decision": "same_account_same_debt",
        "reason": "Records align cleanly.",
        "flags": {"account_match": True, "debt_match": True},
        "at": "2024-07-01T09:30:00Z",
    }

    normalized_decision_a = dict(decision_tag_a)
    normalized_decision_a["flags"] = _normalize_flag_values(decision_tag_a.get("flags", {}))
    normalized_decision_b = dict(decision_tag_b)
    normalized_decision_b["flags"] = _normalize_flag_values(decision_tag_b.get("flags", {}))
    assert normalized_decision_a == {"with": 16, **expected_decision}
    assert normalized_decision_b == {"with": 11, **expected_decision}

    pack_files = sorted(packs_dir.glob("pair_*.jsonl"))
    assert pack_files, "expected at least one AI pack to be written"
    pack_payload = json.loads(pack_files[0].read_text(encoding="utf-8"))
    assert pack_payload["pair"] == {"a": 11, "b": 16}
    assert "decision" not in pack_payload

    result_path = results_dir / pair_result_filename(11, 16)
    assert result_path.exists()
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_payload == {
        "decision": "same_account_same_debt",
        "reason": "Records align cleanly.",
        "flags": {"account_match": True, "debt_match": True},
    }

    assert "ai_result" not in pack_payload

    index_payload = json.loads((base_dir / "index.json").read_text(encoding="utf-8"))
    pairs_entries = index_payload.get("pairs", [])
    matching = [entry for entry in pairs_entries if entry.get("pair") == [11, 16]]
    assert matching
    assert matching[0].get("ai_result") == result_payload
    reverse = [entry for entry in pairs_entries if entry.get("pair") == [16, 11]]
    assert reverse
    assert reverse[0].get("pack_file") == matching[0].get("pack_file")

    pair_tag_a = next(
        tag
        for tag in account_a_tags
        if tag.get("kind") == "same_account_pair" and tag.get("with") == 16
    )
    pair_tag_b = next(
        tag
        for tag in account_b_tags
        if tag.get("kind") == "same_account_pair" and tag.get("with") == 11
    )

    assert pair_tag_a == {
        "kind": "same_account_pair",
        "source": "ai_adjudicator",
        "with": 16,
        "reason": "Records align cleanly.",
        "at": "2024-07-01T09:30:00Z",
    }
    assert pair_tag_b == {
        "kind": "same_account_pair",
        "source": "ai_adjudicator",
        "with": 11,
        "reason": "Records align cleanly.",
        "at": "2024-07-01T09:30:00Z",
    }

    logs_text = logs_path.read_text(encoding="utf-8")
    assert "AI_ADJUDICATOR_RESPONSE" in logs_text


def test_send_ai_merge_packs_writes_same_debt_tags(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runs_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sid = "merge-case"
    pack_payload = {
        "messages": [
            {"role": "system", "content": "instructions"},
            {
                "role": "user",
                "content": "Account 11 and 16 share originator and balance",
            },
        ]
    }
    merge_paths, _ = _create_merge_pack(runs_root, sid, pack_payload)
    packs_dir = merge_paths.packs_dir
    base_dir = merge_paths.base
    results_dir = merge_paths.results_dir
    logs_path = merge_paths.log_file

    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    captured_decisions: list[dict] = []

    def _fake_decide(pack, *, timeout):
        _assert_pack_messages_with_rules(pack, pack_payload)
        captured_decisions.append({"pack": dict(pack), "timeout": timeout})
        return {
            "decision": "same_debt_account_unknown",
            "reason": "Same open date and balance",
            "flags": {"account_match": "unknown", "debt_match": True},
        }

    monkeypatch.setattr(
        send_ai_merge_packs,
        "decide_merge_or_different",
        _fake_decide,
    )
    monkeypatch.setattr(
        send_ai_merge_packs,
        "_isoformat_timestamp",
        lambda dt=None: "2024-06-15T10:00:00Z",
    )

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="scripts.send_ai_merge_packs"):
        send_ai_merge_packs.main(["--sid", sid, "--runs-root", str(runs_root)])

    messages = [record.getMessage() for record in caplog.records]
    assert any(f"MANIFEST_AI_SENT sid={sid}" in message for message in messages)
    assert any(f"MANIFEST_AI_COMPACTED sid={sid}" in message for message in messages)

    stdout = capsys.readouterr().out
    assert "[AI] adjudicated 1 packs (1 success, 0 errors)" in stdout

    assert captured_decisions
    _assert_pack_messages_with_rules(captured_decisions[0]["pack"], pack_payload)

    account_tags_dir = runs_root / sid / "cases" / "accounts"
    tags_a = _normalize_tags(
        json.loads((account_tags_dir / "11" / "tags.json").read_text(encoding="utf-8"))
    )
    tags_b = _normalize_tags(
        json.loads((account_tags_dir / "16" / "tags.json").read_text(encoding="utf-8"))
    )

    expected_decision_tag_a = {
        "kind": "ai_decision",
        "tag": "ai_decision",
        "source": "ai_adjudicator",
        "with": 16,
        "decision": "same_debt_account_unknown",
        "reason": "Same open date and balance",
        "flags": {"account_match": "unknown", "debt_match": True},
        "at": "2024-06-15T10:00:00Z",
    }
    expected_same_debt_tag_a = {
        "kind": "same_debt_pair",
        "source": "ai_adjudicator",
        "with": 16,
        "reason": "Same open date and balance",
        "at": "2024-06-15T10:00:00Z",
    }
    assert tags_a == [expected_decision_tag_a, expected_same_debt_tag_a]

    expected_decision_tag_b = dict(expected_decision_tag_a)
    expected_decision_tag_b["with"] = 11
    expected_same_debt_tag_b = dict(expected_same_debt_tag_a)
    expected_same_debt_tag_b["with"] = 11
    assert tags_b == [expected_decision_tag_b, expected_same_debt_tag_b]

    log_lines = logs_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("AI_ADJUDICATOR_PACK_START" in line for line in log_lines)
    assert any("AI_ADJUDICATOR_REQUEST" in line for line in log_lines)
    assert any("AI_ADJUDICATOR_RESPONSE" in line for line in log_lines)
    assert any("AI_ADJUDICATOR_PACK_SUCCESS" in line for line in log_lines)

    manifest = RunManifest.for_sid(sid)
    manifest_data = json.loads((runs_root / sid / "manifest.json").read_text(encoding="utf-8"))
    ai_packs = manifest_data["ai"]["packs"]
    assert Path(ai_packs["dir"]) == base_dir.resolve()
    assert Path(ai_packs["dir"]).exists()
    assert Path(ai_packs["packs_dir"]) == (base_dir / "packs").resolve()
    assert Path(ai_packs["results_dir"]) == (base_dir / "results").resolve()
    assert Path(ai_packs["index"]) == (base_dir / "index.json").resolve()
    assert Path(ai_packs["index"]).exists()
    assert Path(ai_packs["logs"]) == logs_path.resolve()
    assert ai_packs.get("pairs", 0) >= 1

    status_info = manifest.data.get("ai", {}).get("status", {})
    assert status_info.get("sent") is True
    assert status_info.get("compacted") is True
    assert status_info.get("skipped_reason") is None

    result_path = results_dir / pair_result_filename(11, 16)
    assert result_path.exists()


@pytest.mark.parametrize(
    "decision,flags,expected_pair",
    [
        (
            "same_account_diff_debt",
            {"account_match": True, "debt_match": False},
            "same_account_pair",
        ),
        (
            "same_debt_diff_account",
            {"account_match": False, "debt_match": True},
            "same_debt_pair",
        ),
        (
            "same_account_debt_unknown",
            {"account_match": True, "debt_match": "unknown"},
            "same_account_pair",
        ),
        (
            "same_debt_account_unknown",
            {"account_match": "unknown", "debt_match": True},
            "same_debt_pair",
        ),
        (
            "different",
            {"account_match": False, "debt_match": False},
            None,
        ),
    ],
)
def test_send_ai_merge_packs_writes_decision_variants(
    monkeypatch: pytest.MonkeyPatch,
    runs_root: Path,
    decision: str,
    flags: dict[str, bool | str],
    expected_pair: str | None,
) -> None:
    sid = f"variant-{decision}"
    pack_payload = {
        "messages": [
            {"role": "system", "content": "instructions"},
            {
                "role": "user",
                "content": "Account 11 and 16 require adjudication",
            },
        ]
    }
    merge_paths, _ = _create_merge_pack(runs_root, sid, pack_payload)

    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    reason = f"Reason for {decision}"
    timestamp = "2024-06-20T12:00:00Z"

    def _fake_decide(pack: dict, *, timeout: float):
        _assert_pack_messages_with_rules(pack, pack_payload)
        return {"decision": decision, "reason": reason, "flags": dict(flags)}

    monkeypatch.setattr(send_ai_merge_packs, "decide_merge_or_different", _fake_decide)
    monkeypatch.setattr(
        send_ai_merge_packs,
        "_isoformat_timestamp",
        lambda dt=None: timestamp,
    )

    send_ai_merge_packs.main(["--sid", sid, "--runs-root", str(runs_root)])

    result_path = merge_paths.results_dir / pair_result_filename(11, 16)
    assert result_path.exists()

    account_tags_dir = runs_root / sid / "cases" / "accounts"
    tags_a = _normalize_tags(
        json.loads((account_tags_dir / "11" / "tags.json").read_text(encoding="utf-8"))
    )
    tags_b = _normalize_tags(
        json.loads((account_tags_dir / "16" / "tags.json").read_text(encoding="utf-8"))
    )

    def _expected_flags(raw_flags: dict[str, bool | str]) -> dict[str, bool | str]:
        normalized: dict[str, bool | str] = {}
        for key, value in raw_flags.items():
            if isinstance(value, str):
                lowered = value.strip().lower()
                normalized[key] = "unknown" if lowered == "unknown" else lowered == "true"
            else:
                normalized[key] = bool(value)
        return normalized

    expected_flags = _expected_flags(flags)
    expected_decision_tag = {
        "kind": "ai_decision",
        "tag": "ai_decision",
        "source": "ai_adjudicator",
        "with": 16,
        "decision": decision,
        "reason": reason,
        "flags": expected_flags,
        "at": timestamp,
    }
    expected_tags_a = [expected_decision_tag]
    if expected_pair is not None:
        expected_tags_a.append(
            {
                "kind": expected_pair,
                "source": "ai_adjudicator",
                "with": 16,
                "reason": reason,
                "at": timestamp,
            }
        )

    mirrored_decision = dict(expected_decision_tag)
    mirrored_decision["with"] = 11
    mirrored_pair = None
    if expected_pair is not None:
        mirrored_pair = {
            "kind": expected_pair,
            "source": "ai_adjudicator",
            "with": 11,
            "reason": reason,
            "at": timestamp,
        }

    assert tags_a == expected_tags_a
    expected_tags_b = [mirrored_decision]
    if mirrored_pair is not None:
        expected_tags_b.append(mirrored_pair)
    assert tags_b == expected_tags_b
def test_send_ai_merge_packs_logs_failure(monkeypatch: pytest.MonkeyPatch, runs_root: Path) -> None:
    sid = "merge-case-errors"
    pack_payload = {
        "messages": [
            {"role": "system", "content": "instructions"},
            {"role": "user", "content": "Account pair"},
        ]
    }
    merge_paths, _ = _create_merge_pack(runs_root, sid, pack_payload, a_idx=1, b_idx=2)
    packs_dir = merge_paths.packs_dir
    logs_path = merge_paths.log_file

    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("AI_MAX_RETRIES", "0")

    def _fake_decide(pack, *, timeout):
        _assert_pack_messages_with_rules(pack, pack_payload)
        return {"decision": "maybe", "reason": ""}

    monkeypatch.setattr(send_ai_merge_packs, "decide_merge_or_different", _fake_decide)
    monkeypatch.setattr(
        send_ai_merge_packs,
        "_isoformat_timestamp",
        lambda dt=None: "2024-06-15T12:00:00Z",
    )

    with pytest.raises(SystemExit):
        send_ai_merge_packs.main(["--sid", sid, "--runs-root", str(runs_root)])

    log_lines = logs_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("AI_ADJUDICATOR_PACK_START" in line for line in log_lines)
    assert any("AI_ADJUDICATOR_REQUEST" in line for line in log_lines)
    assert any("AI_ADJUDICATOR_RESPONSE" in line for line in log_lines)
    assert any("AI_ADJUDICATOR_ERROR" in line for line in log_lines)
    assert any("AI_ADJUDICATOR_PACK_FAILURE" in line for line in log_lines)


def test_send_ai_merge_packs_reads_legacy_layout(
    monkeypatch: pytest.MonkeyPatch,
    runs_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sid = "legacy-layout"
    legacy_dir = runs_root / sid / "ai_packs"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    pack_payload = {
        "messages": [
            {"role": "system", "content": "legacy instructions"},
            {"role": "user", "content": "Legacy merge pack"},
        ]
    }
    pack_filename = pair_pack_filename(11, 16)
    (legacy_dir / pack_filename).write_text(
        json.dumps(pack_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    index_payload = {
        "sid": sid,
        "pairs": [
            {
                "a": 11,
                "b": 16,
                "pair": [11, 16],
                "pack_file": pack_filename,
                "lines_a": 0,
                "lines_b": 0,
                "score_total": 0,
            }
        ],
    }
    (legacy_dir / "index.json").write_text(
        json.dumps(index_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    def _fake_decide(pack: dict, *, timeout: float):
        _assert_pack_messages_with_rules(pack, pack_payload)
        return {
            "decision": "same_account_same_debt",
            "reason": "Legacy directories are supported.",
            "flags": {"account_match": True, "debt_match": True},
        }

    monkeypatch.setattr(send_ai_merge_packs, "decide_merge_or_different", _fake_decide)
    monkeypatch.setattr(
        send_ai_merge_packs,
        "_isoformat_timestamp",
        lambda dt=None: "2024-07-10T15:30:00Z",
    )

    merge_paths = get_merge_paths(runs_root, sid, create=False)

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="scripts.send_ai_merge_packs"):
        send_ai_merge_packs.main(["--sid", sid, "--runs-root", str(runs_root)])

    messages = [record.getMessage() for record in caplog.records]
    assert any(f"SENDER_PACKS_DIR_LEGACY sid={sid}" in message for message in messages)

    result_dir = merge_paths.results_dir
    assert result_dir.exists()
    result_path = result_dir / pair_result_filename(11, 16)
    assert result_path.exists()
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert result_payload["decision"] == "same_account_same_debt"

    pack_path = legacy_dir / pack_filename
    updated_payload = json.loads(pack_path.read_text(encoding="utf-8"))
    assert "ai_result" not in updated_payload

    manifest = RunManifest.for_sid(sid)
    packs_info = manifest.data.get("ai", {}).get("packs", {})
    assert Path(packs_info.get("dir")) == merge_paths.base.resolve()
    assert Path(packs_info.get("index")).exists()
    assert Path(packs_info.get("logs")).exists()

    logs_path = merge_paths.log_file
    assert logs_path.exists()
    log_lines = logs_path.read_text(encoding="utf-8").splitlines()
    assert any("AI_ADJUDICATOR_RESPONSE" in line for line in log_lines)


def test_write_decision_tags_idempotent(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    sid = "case-020"
    runs_root.mkdir(parents=True, exist_ok=True)

    reason = "Same debt - Aligned balances"

    send_ai_merge_packs._write_decision_tags(
        runs_root,
        sid,
        31,
        32,
        "same_debt_account_unknown",
        reason,
        "2024-07-01T00:00:00Z",
        {"decision": "same_debt_account_unknown", "reason": reason},
    )

    # Second invocation should update without duplication.
    send_ai_merge_packs._write_decision_tags(
        runs_root,
        sid,
        31,
        32,
        "same_debt_account_unknown",
        reason,
        "2024-07-01T00:00:00Z",
        {"decision": "same_debt_account_unknown", "reason": reason},
    )

    base = runs_root / sid / "cases" / "accounts"
    tags_a = _normalize_tags(
        json.loads((base / "31" / "tags.json").read_text(encoding="utf-8"))
    )
    tags_b = _normalize_tags(
        json.loads((base / "32" / "tags.json").read_text(encoding="utf-8"))
    )

    expected_decision_a = {
        "kind": "ai_decision",
        "tag": "ai_decision",
        "source": "ai_adjudicator",
        "with": 32,
        "decision": "same_debt_account_unknown",
        "reason": reason,
        "at": "2024-07-01T00:00:00Z",
    }
    expected_same_debt_a = {
        "kind": "same_debt_pair",
        "with": 32,
        "source": "ai_adjudicator",
        "reason": reason,
        "at": "2024-07-01T00:00:00Z",
    }

    expected_decision_b = dict(expected_decision_a)
    expected_decision_b["with"] = 31
    expected_same_debt_b = dict(expected_same_debt_a)
    expected_same_debt_b["with"] = 31

    assert tags_a == [expected_decision_a, expected_same_debt_a]
    assert tags_b == [expected_decision_b, expected_same_debt_b]


def test_ai_pairing_flow_compaction(
    monkeypatch: pytest.MonkeyPatch, runs_root: Path
) -> None:
    sid = "codex-flow"
    accounts_root = runs_root / sid / "cases" / "accounts"
    account_a_dir = accounts_root / "11"
    account_b_dir = accounts_root / "16"

    _write_raw_lines(
        account_a_dir / "raw_lines.json",
        [
            "US BK CACS",
            "Account # 409451******",
            "Balance Owed: $12,091",
            "Last Payment: 2024-02-11",
            "Creditor Remarks: Transferred",
        ],
    )
    _write_raw_lines(
        account_b_dir / "raw_lines.json",
        [
            "U S BANK",
            "Account # -- 409451******",
            "Balance Owed: -- $12,091 --",
            "Last Payment: 2024-02-11",
            "Remarks: Referred to collections",
        ],
    )

    _write_json(account_a_dir / "tags.json", [_merge_best_tag(16)])
    _write_json(account_b_dir / "tags.json", [_merge_best_tag(11)])

    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))
    build_ai_merge_packs_main(
        ["--sid", sid, "--runs-root", str(runs_root), "--max-lines-per-side", "6"]
    )

    merge_paths = get_merge_paths(runs_root, sid, create=False)
    packs_dir = merge_paths.packs_dir
    assert merge_paths.index_file.exists()

    timestamp = "2024-07-04T12:00:00Z"

    def _fake_decide(pack, *, timeout):
        assert pack["pair"] == {"a": 11, "b": 16}
        return {
            "decision": "different",
            "reason": "These tradelines describe the same debt from different collectors.",
            "flags": {"account_match": False, "debt_match": False},
        }

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(send_ai_merge_packs, "decide_merge_or_different", _fake_decide)
    monkeypatch.setattr(send_ai_merge_packs, "_isoformat_timestamp", lambda dt=None: timestamp)

    send_ai_merge_packs.main(["--sid", sid, "--runs-root", str(runs_root)])

    result_path = merge_paths.results_dir / pair_result_filename(11, 16)
    assert result_path.exists()

    for account_dir in (account_a_dir, account_b_dir):
        compact_tags_for_account(account_dir)

    tags_a = _normalize_tags(
        json.loads((account_a_dir / "tags.json").read_text(encoding="utf-8"))
    )
    tags_b = _normalize_tags(
        json.loads((account_b_dir / "tags.json").read_text(encoding="utf-8"))
    )

    assert tags_a == [
        {"kind": "merge_best", "with": 16, "decision": "ai"},
        {
            "kind": "ai_decision",
            "with": 16,
            "decision": "different",
            "flags": {"account_match": False, "debt_match": False},
            "at": timestamp,
        },
        {
            "kind": "ai_resolution",
            "with": 16,
            "decision": "different",
            "flags": {"account_match": False, "debt_match": False},
            "reason": "These tradelines describe the same debt from different collectors.",
        },
    ]
    assert tags_b == [
        {"kind": "merge_best", "with": 11, "decision": "ai"},
        {
            "kind": "ai_decision",
            "with": 11,
            "decision": "different",
            "flags": {"account_match": False, "debt_match": False},
            "at": timestamp,
        },
        {
            "kind": "ai_resolution",
            "with": 11,
            "decision": "different",
            "flags": {"account_match": False, "debt_match": False},
            "reason": "These tradelines describe the same debt from different collectors.",
        },
    ]

    summary_a = json.loads((account_a_dir / "summary.json").read_text(encoding="utf-8"))
    summary_b = json.loads((account_b_dir / "summary.json").read_text(encoding="utf-8"))

    def _ai_summary(entry_list: list[dict[str, object]], *, partner: int) -> dict[str, object]:
        return next(item for item in entry_list if item.get("with") == partner)

    ai_a = summary_a["ai_explanations"]
    kinds_a = {item["kind"] for item in ai_a}
    assert kinds_a == {"ai_decision", "ai_resolution", "same_account_pair"}
    ai_entry_a = next(item for item in ai_a if item.get("kind") == "ai_decision")
    assert ai_entry_a["with"] == 16
    assert ai_entry_a.get("normalized") is False
    assert "reason" in ai_entry_a
    assert "same debt" in ai_entry_a["reason"].lower()
    resolution_entry_a = next(
        item for item in ai_a if item.get("kind") == "ai_resolution"
    )
    assert resolution_entry_a["with"] == 16
    assert resolution_entry_a.get("normalized") is False
    assert resolution_entry_a.get("flags") == {
        "account_match": False,
        "debt_match": False,
    }
    assert "same debt" in resolution_entry_a.get("reason", "").lower()
    pair_entry_a = next(item for item in ai_a if item.get("kind") == "same_account_pair")
    assert pair_entry_a["with"] == 16
    assert "same debt" in pair_entry_a.get("reason", "").lower()

    ai_b = summary_b["ai_explanations"]
    kinds_b = {item["kind"] for item in ai_b}
    assert kinds_b == {"ai_decision", "ai_resolution", "same_account_pair"}
    ai_entry_b = next(item for item in ai_b if item.get("kind") == "ai_decision")
    assert ai_entry_b["with"] == 11
    assert ai_entry_b.get("normalized") is False
    assert "reason" in ai_entry_b
    assert "same debt" in ai_entry_b["reason"].lower()
    resolution_entry_b = next(
        item for item in ai_b if item.get("kind") == "ai_resolution"
    )
    assert resolution_entry_b["with"] == 11
    assert resolution_entry_b.get("normalized") is False
    assert resolution_entry_b.get("flags") == {
        "account_match": False,
        "debt_match": False,
    }
    assert "same debt" in resolution_entry_b.get("reason", "").lower()
    pair_entry_b = next(item for item in ai_b if item.get("kind") == "same_account_pair")
    assert pair_entry_b["with"] == 11
    assert "same debt" in pair_entry_b.get("reason", "").lower()

    merge_summary_a = summary_a.get("merge_explanations", [])
    assert merge_summary_a
    merge_best_entry = _ai_summary(merge_summary_a, partner=16)
    assert merge_best_entry["kind"] == "merge_best"
    assert merge_best_entry["parts"] == {"balance_owed": 31, "account_number": 28}
    assert merge_best_entry["conflicts"] == ["credit_limit:conflict"]
    assert merge_best_entry["matched_fields"] == {
        "balance_owed": True,
        "last_payment": True,
        "account_number": True,
    }
    assert merge_best_entry.get("acctnum_level") in {"exact_or_known_match", "none"}
    acct_pair = merge_best_entry["matched_pairs"]["account_number"]
    assert isinstance(acct_pair, list)
    assert len(acct_pair) == 2
    assert "acctnum_digits_len_a" in merge_best_entry
    assert "acctnum_digits_len_b" in merge_best_entry

    merge_pair_entries = [
        entry for entry in merge_summary_a if entry.get("kind") == "merge_pair"
    ]
    assert merge_pair_entries
    merge_pair_entry = merge_pair_entries[0]
    assert merge_pair_entry["with"] == 16
    assert merge_pair_entry["decision"] == "ai"
    assert merge_pair_entry.get("total", 0) >= 0

    merge_summary_b = summary_b.get("merge_explanations", [])
    assert merge_summary_b
    merge_best_entry_b = _ai_summary(merge_summary_b, partner=11)
    assert merge_best_entry_b["kind"] == "merge_best"
    assert merge_best_entry_b["with"] == 11

    merge_score_a = summary_a.get("merge_scoring")
    assert merge_score_a
    assert merge_score_a["best_with"] == 16
    assert merge_score_a["score_total"] >= 0
    assert merge_score_a["matched_fields"].get("balance_owed") is True
    assert merge_score_a.get("acctnum_level") in {"exact_or_known_match", "none"}
    assert "matched_pairs" in merge_score_a
    assert "account_number" in merge_score_a["matched_pairs"]
    assert isinstance(merge_score_a["matched_pairs"]["account_number"], list)
    assert "acctnum_digits_len_a" in merge_score_a
    assert "acctnum_digits_len_b" in merge_score_a

    merge_score_b = summary_b.get("merge_scoring")
    assert merge_score_b
    assert merge_score_b["best_with"] == 11
    assert merge_score_b["matched_fields"].get("balance_owed") is True
    assert merge_score_b.get("acctnum_level") in {"exact_or_known_match", "none"}
    assert "matched_pairs" in merge_score_b
    assert "account_number" in merge_score_b["matched_pairs"]
    assert isinstance(merge_score_b["matched_pairs"]["account_number"], list)
    assert "acctnum_digits_len_a" in merge_score_b
    assert "acctnum_digits_len_b" in merge_score_b

    # Tags should remain compact on repeated compaction.
    snapshot_a = json.loads((account_a_dir / "tags.json").read_text(encoding="utf-8"))
    compact_tags_for_account(account_a_dir)
    assert json.loads((account_a_dir / "tags.json").read_text(encoding="utf-8")) == snapshot_a
