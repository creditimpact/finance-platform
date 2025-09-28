import json
from pathlib import Path
from typing import Mapping

import pytest

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
                "transunion": {"account_number_display": f"{base_digits[:6]}****{base_digits[-4:]}"},
                "experian": {"account_number": f"****{base_digits[-4:]}"},
            },
            "raw_lines": [
                "Creditor A",
                f"Account # {base_digits[:6]}****{base_digits[-4:]} -- ****{base_digits[-4:]} -- {base_digits}",
            ],
        },
        15: {
            "bureaus": {
                "experian": {"account_number_display": base_digits},
                "equifax": {"account_number": f"XXXX{base_digits[-4:]}"},
            },
            "raw_lines": [
                "Creditor B",
                f"Account # {base_digits} -- XXXX{base_digits[-4:]}",
            ],
        },
        29: {
            "bureaus": {
                "transunion": {"account_number": base_digits},
                "equifax": {"account_number_display": f"****{base_digits[-4:]}"},
            },
            "raw_lines": [
                "Creditor C",
                f"Account # ****{base_digits[-4:]} -- {base_digits}",
            ],
        },
        39: {
            "bureaus": {
                "experian": {"account_number": f"{base_digits[:4]}****{base_digits[-4:]}"},
                "equifax": {"account_number_display": base_digits},
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

    packs_dir = runs_root / sid / "ai_packs"
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


def test_merge_decision_alias_members() -> None:
    assert MergeDecision.SAME_ACCOUNT_SAME_DEBT is MergeDecision.MERGE
    assert MergeDecision.SAME_DEBT_ACCOUNT_UNKNOWN is MergeDecision.SAME_DEBT
