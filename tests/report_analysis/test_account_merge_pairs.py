import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, cast

import pytest

from backend.core.ai.paths import get_merge_paths
from backend.core.io.tags import upsert_tag
from backend.core.logic.report_analysis.account_merge import (
    AI_PACK_SCORE_THRESHOLD,
    MergeDecision,
    build_merge_pair_tag,
    build_summary_ai_entries,
    choose_best_partner,
    gen_unordered_pairs,
    persist_merge_tags,
    score_all_pairs_0_100,
)
from backend.config import merge_config
from backend.pipeline.runs import RunManifest, persist_manifest
from scripts.build_ai_merge_packs import main as build_packs_main


def test_gen_unordered_pairs_sorted_unique_pairs() -> None:
    indices = [16, 11, 8, 11, 12]

    result = gen_unordered_pairs(indices)

    assert result == [
        (8, 11),
        (8, 12),
        (8, 16),
        (11, 12),
        (11, 16),
        (12, 16),
    ]


def _write_account_payload(
    accounts_root: Path,
    idx: int,
    bureaus: dict[str, object],
    raw_lines: list[str],
) -> None:
    account_dir = accounts_root / str(idx)
    account_dir.mkdir(parents=True, exist_ok=True)

    bureaus_path = account_dir / "bureaus.json"
    bureaus_path.write_text(
        json.dumps(bureaus, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_path = account_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"account_index": idx}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raw_payload = [{"text": line} for line in raw_lines]
    raw_lines_path = account_dir / "raw_lines.json"
    raw_lines_path.write_text(
        json.dumps(raw_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    tags_path = account_dir / "tags.json"
    tags_path.write_text("[]\n", encoding="utf-8")


_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "backend" / "tests" / "fixtures" / "account_merge"


def _load_fixture(name: str) -> dict[str, Any]:
    fixture_path = _FIXTURE_DIR / name
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _strip_original_creditor(payload: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = cast(dict[str, Any], deepcopy(payload))
    for bureau in ("transunion", "experian", "equifax"):
        section = sanitized.get(bureau)
        if isinstance(section, dict):
            section.pop("original_creditor", None)
    return sanitized


def test_account_number_clique_persists_all_pair_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID-CLIQUE"
    runs_root = tmp_path / "runs"
    accounts_root = runs_root / sid / "cases" / "accounts"

    base_digits = "4094517890"

    account_specs = {
        14: {
            "bureaus": {
                "transunion": {
                    "account_number_display": f"{base_digits[:6]}****{base_digits[-4:]}",
                    "original_creditor": "Creditor A",
                },
                "experian": {
                    "account_number": f"****{base_digits[-4:]}",
                    "original_creditor": "Creditor A",
                },
            },
            "raw_lines": [
                "Creditor A",

                f"Account # {base_digits[:6]}****{base_digits[-4:]} -- ****{base_digits[-4:]} -- {base_digits}",
            ],
        },
        15: {
            "bureaus": {
                "experian": {
                    "account_number_display": base_digits,
                    "original_creditor": "Creditor B",
                },
                "equifax": {
                    "account_number": f"XXXX{base_digits[-4:]}",
                    "original_creditor": "Creditor B",
                },
            },
            "raw_lines": [
                "Creditor B",
                f"Account # {base_digits} -- XXXX{base_digits[-4:]}",
            ],
        },
        29: {
            "bureaus": {
                "transunion": {
                    "account_number": base_digits,
                    "original_creditor": "Creditor C",
                },
                "equifax": {
                    "account_number_display": f"****{base_digits[-4:]}",
                    "original_creditor": "Creditor C",
                },
            },
            "raw_lines": [
                "Creditor C",
                f"Account # ****{base_digits[-4:]} -- {base_digits}",
            ],
        },
        39: {
            "bureaus": {
                "experian": {
                    "account_number": f"{base_digits[:4]}****{base_digits[-4:]}",
                    "original_creditor": "Creditor D",
                },
                "equifax": {
                    "account_number_display": base_digits,
                    "original_creditor": "Creditor D",
                },
            },
            "raw_lines": [
                "Creditor D",
                f"Account # {base_digits[:4]}****{base_digits[-4:]} -- {base_digits}",
            ],
        },
    }

    for idx, spec in account_specs.items():
        _write_account_payload(accounts_root, idx, spec["bureaus"], spec["raw_lines"])

    monkeypatch.setenv("AI_THRESHOLD", "27")
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))

    indices = sorted(account_specs.keys())
    scores = score_all_pairs_0_100(sid, indices, runs_root=runs_root)
    best = choose_best_partner(scores)
    simplified = {
        str(left): {
            str(right): {
                "decision": str(details.get("decision")),
                "total": details.get("total"),
            }
            for right, details in partners.items()
        }
        for left, partners in scores.items()
    }
    print("MERGE_SCORES", simplified)
    persist_merge_tags(sid, scores, best, runs_root=runs_root)

    for left, partner_map in scores.items():
        for right, result in partner_map.items():
            if right <= left:
                continue
            decision = str(result.get("decision", "")).lower()
            if decision not in {"ai", "auto"}:
                continue
            left_tag = build_merge_pair_tag(right, result)
            right_tag = build_merge_pair_tag(left, result)
            upsert_tag(accounts_root / str(left), left_tag, unique_keys=("kind", "with"))
            upsert_tag(accounts_root / str(right), right_tag, unique_keys=("kind", "with"))

    build_packs_main(
        [
            "--sid",
            sid,
            "--runs-root",
            str(runs_root),
            "--include-all-pairs",
            "--max-lines-per-side",
            "5",
        ]
    )

    expected_pairs = [
        (14, 15),
        (14, 29),
        (14, 39),
        (15, 29),
        (15, 39),
        (29, 39),
    ]

    merge_paths = get_merge_paths(runs_root, sid, create=False)
    packs_dir = merge_paths.packs_dir
    for left, right in expected_pairs:
        first, second = sorted((left, right))
        pack_path = packs_dir / f"pair_{first:03d}_{second:03d}.jsonl"
        assert pack_path.exists(), f"Missing pack for pair {(first, second)}"

        lines = [
            json.loads(line)
            for line in pack_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert lines, f"Pack {pack_path} is empty"
        pack_payload = lines[0]
        highlights = pack_payload.get("highlights", {})
        assert highlights.get("acctnum_level") == "exact_or_known_match"

    for idx in indices:
        summary_path = accounts_root / str(idx) / "summary.json"
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        merge_entries = summary_payload.get("merge_explanations", [])
        pair_partners = {
            entry.get("with")
            for entry in merge_entries
            if isinstance(entry, Mapping) and entry.get("kind") == "merge_pair"
        }
        expected_partners = {
            partner
            for partner, result in (scores.get(idx, {}) or {}).items()
            if isinstance(result, Mapping)
            and int(result.get("total") or 0) >= AI_PACK_SCORE_THRESHOLD
        }
        assert expected_partners <= pair_partners


def test_build_summary_ai_entries_normalizes_aliases() -> None:
    entries = build_summary_ai_entries(
        7,
        "same_debt",
        "Reasoning",
        {"account_match": "unknown", "debt_match": True},
    )

    ai_entry = next(entry for entry in entries if entry.get("kind") == "ai_decision")
    assert ai_entry["decision"] == "same_debt_account_unknown"
    assert ai_entry["normalized"] is True
    assert ai_entry["ai_result"]["decision"] == "same_debt_account_unknown"

    resolution_entry = next(
        entry for entry in entries if entry.get("kind") == "ai_resolution"
    )
    assert resolution_entry["decision"] == "same_debt_account_unknown"
    assert resolution_entry["normalized"] is True
    assert resolution_entry["ai_result"]["decision"] == "same_debt_account_unknown"

    pair_entry = next(entry for entry in entries if entry.get("kind") == "same_debt_pair")
    assert pair_entry["ai_result"]["decision"] == "same_debt_account_unknown"
    assert "same_debt_account_unclear" in pair_entry.get("notes", [])


def test_build_packs_index_records_skip_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID-SKIPS"
    runs_root = tmp_path / "runs"
    accounts_root = runs_root / sid / "cases" / "accounts"

    left_payload = _strip_original_creditor(_load_fixture("trimmed_9.json"))
    right_payload = _strip_original_creditor(_load_fixture("trimmed_10.json"))

    for idx, payload in enumerate((left_payload, right_payload), start=1):
        _write_account_payload(
            accounts_root,
            idx,
            payload,
            [f"Fixture account {idx}"],
        )

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("AI_THRESHOLD", "0")
    monkeypatch.setenv("MERGE_REQUIRE_ORIGINAL_CREDITOR_FOR_AI", "1")
    monkeypatch.setenv("MERGE_ZERO_PACKS_SIGNAL", "1")
    monkeypatch.setenv("MERGE_SKIP_COUNTS_ENABLED", "1")
    monkeypatch.setenv("MERGE_ENABLED", "1")
    monkeypatch.setenv("MERGE_POINTS_MODE", "1")
    monkeypatch.setenv("MERGE_AI_POINTS_THRESHOLD", "3")
    monkeypatch.setenv("MERGE_DIRECT_POINTS_THRESHOLD", "99")
    merge_config.reset_merge_config_cache()

    try:
        indices = [1, 2]
        scores = score_all_pairs_0_100(sid, indices, runs_root=runs_root)
        best = choose_best_partner(scores)
        persist_merge_tags(sid, scores, best, runs_root=runs_root)

        manifest_path = runs_root / sid / "manifest.json"
        manifest = RunManifest.load_or_create(manifest_path, sid, allow_create=True)
        persist_manifest(manifest)

        build_packs_main(
            [
                "--sid",
                sid,
                "--runs-root",
                str(runs_root),
                "--include-all-pairs",
                "--max-lines-per-side",
                "5",
            ]
        )

        merge_paths = get_merge_paths(runs_root, sid, create=False)
        index_path = merge_paths.index_file
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        totals = payload.get("totals", {})

        assert totals.get("merge_zero_packs") is True
        skip_counts = totals.get("skip_counts") or {}
        assert skip_counts.get("missing_original_creditor") == 1
        assert totals.get("skip_reason_top") == "missing_original_creditor"
        assert totals.get("packs_built") == 0
        assert totals.get("created_packs") == 0
    finally:
        merge_config.reset_merge_config_cache()


def test_merge_zero_packs_emits_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID-TELEM"
    runs_root = tmp_path / "runs"
    accounts_root = runs_root / sid / "cases" / "accounts"

    left_payload = _strip_original_creditor(_load_fixture("trimmed_9.json"))
    right_payload = _strip_original_creditor(_load_fixture("trimmed_10.json"))

    for idx, payload in enumerate((left_payload, right_payload), start=1):
        _write_account_payload(
            accounts_root,
            idx,
            payload,
            [f"Fixture account {idx}"],
        )

    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("AI_THRESHOLD", "0")
    monkeypatch.setenv("MERGE_REQUIRE_ORIGINAL_CREDITOR_FOR_AI", "1")
    monkeypatch.setenv("MERGE_ZERO_PACKS_SIGNAL", "1")
    monkeypatch.setenv("MERGE_SKIP_COUNTS_ENABLED", "1")
    monkeypatch.setenv("MERGE_ENABLED", "1")
    monkeypatch.setenv("MERGE_POINTS_MODE", "1")
    monkeypatch.setenv("MERGE_AI_POINTS_THRESHOLD", "3")
    monkeypatch.setenv("MERGE_DIRECT_POINTS_THRESHOLD", "99")
    merge_config.reset_merge_config_cache()

    captured: list[tuple[str, object]] = []

    def _capture_counter(name: str, increment: object = 1) -> None:
        captured.append((name, increment))

    monkeypatch.setattr(
        "backend.core.logic.report_analysis.account_merge.emit_counter",
        _capture_counter,
    )

    try:
        indices = [1, 2]
        scores = score_all_pairs_0_100(sid, indices, runs_root=runs_root)
        best = choose_best_partner(scores)
        persist_merge_tags(sid, scores, best, runs_root=runs_root)

        manifest_path = runs_root / sid / "manifest.json"
        manifest = RunManifest.load_or_create(manifest_path, sid, allow_create=True)
        persist_manifest(manifest)

        build_packs_main(
            [
                "--sid",
                sid,
                "--runs-root",
                str(runs_root),
                "--include-all-pairs",
                "--max-lines-per-side",
                "5",
            ]
        )
    finally:
        merge_config.reset_merge_config_cache()

    zero_pack_metrics = [entry for entry in captured if entry[0].startswith("merge.zero_packs")]
    assert zero_pack_metrics, f"expected zero-pack telemetry, got {captured!r}"

    total_calls = [increment for name, increment in captured if name == "merge.zero_packs.total"]
    assert total_calls == [1]

    reason_metric = "merge.zero_packs.reason.missing_original_creditor"
    reason_calls = [increment for name, increment in captured if name == reason_metric]
    assert reason_calls and sum(int(value) for value in reason_calls) >= 1


def test_merge_decision_alias_members() -> None:
    assert MergeDecision.SAME_ACCOUNT_SAME_DEBT is MergeDecision.MERGE
    assert MergeDecision.SAME_DEBT_ACCOUNT_UNKNOWN is MergeDecision.SAME_DEBT
