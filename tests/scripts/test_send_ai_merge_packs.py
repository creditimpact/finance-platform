import json
from pathlib import Path

import pytest

from backend.core.logic.report_analysis import ai_sender
from scripts import send_ai_merge_packs


@pytest.fixture()
def runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_send_ai_merge_packs_writes_same_debt_tags(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], runs_root: Path
) -> None:
    sid = "merge-case"
    packs_dir = runs_root / sid / "ai_packs"
    packs_dir.mkdir(parents=True, exist_ok=True)

    pack_payload = {
        "messages": [
            {"role": "system", "content": "instructions"},
            {
                "role": "user",
                "content": "Account 11 and 16 share originator and balance",
            },
        ]
    }
    pack_path = packs_dir / "pair-11-16.json"
    pack_path.write_text(json.dumps(pack_payload), encoding="utf-8")
    index_payload = [{"a": 11, "b": 16, "file": pack_path.name}]
    (packs_dir / "index.json").write_text(json.dumps(index_payload), encoding="utf-8")

    monkeypatch.setenv("ENABLE_AI_ADJUDICATOR", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    captured_attempts: list[dict] = []

    def _fake_send(pack, config, request=None):
        captured_attempts.append({"pack": dict(pack), "model": config.model})
        assert pack == pack_payload
        return "same_debt", "Same open date and balance"

    monkeypatch.setattr(ai_sender, "send_single_attempt", _fake_send)
    monkeypatch.setattr(ai_sender, "isoformat_timestamp", lambda dt=None: "2024-06-15T10:00:00Z")

    send_ai_merge_packs.main(["--sid", sid, "--runs-root", str(runs_root)])

    stdout = capsys.readouterr().out
    assert "[AI] adjudicated 1 packs (1 success, 0 errors)" in stdout

    assert captured_attempts
    assert captured_attempts[0]["pack"] == pack_payload

    account_tags_dir = runs_root / sid / "cases" / "accounts"
    tags_a = json.loads((account_tags_dir / "11" / "tags.json").read_text(encoding="utf-8"))
    tags_b = json.loads((account_tags_dir / "16" / "tags.json").read_text(encoding="utf-8"))

    expected_decision_tag_a = {
        "kind": "ai_decision",
        "tag": "ai_decision",
        "source": "ai_adjudicator",
        "with": 16,
        "decision": "same_debt",
        "reason": "Same open date and balance",
        "at": "2024-06-15T10:00:00Z",
    }
    expected_same_debt_tag_a = {
        "kind": "same_debt_pair",
        "source": "ai_adjudicator",
        "with": 16,
        "at": "2024-06-15T10:00:00Z",
    }
    assert tags_a == [expected_decision_tag_a, expected_same_debt_tag_a]

    expected_decision_tag_b = dict(expected_decision_tag_a)
    expected_decision_tag_b["with"] = 11
    expected_same_debt_tag_b = dict(expected_same_debt_tag_a)
    expected_same_debt_tag_b["with"] = 11
    assert tags_b == [expected_decision_tag_b, expected_same_debt_tag_b]

    logs_path = packs_dir / "logs.txt"
    log_lines = logs_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("AI_ADJUDICATOR_REQUEST" in line for line in log_lines)
    assert any("AI_ADJUDICATOR_RESPONSE" in line for line in log_lines)
    assert any("AI_ADJUDICATOR_PACK_SUCCESS" in line for line in log_lines)

    manifest_path = runs_root / sid / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ai_artifacts = manifest["artifacts"]["ai"]
    assert Path(ai_artifacts["packs_dir"]) == packs_dir.resolve()
    assert Path(ai_artifacts["packs_index"]) == (packs_dir / "index.json").resolve()
    assert Path(ai_artifacts["logs"]) == logs_path.resolve()
