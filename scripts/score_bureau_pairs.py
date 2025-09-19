"""CLI for inspecting deterministic account merge scores between bureau accounts."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from backend.core.logic.report_analysis.account_merge import (
    _FIELD_SEQUENCE,
    _build_aux_payload,
    _merge_tag_from_best,
    _sanitize_parts,
    choose_best_partner,
    get_merge_cfg,
    load_bureaus,
    score_pair,
)


DEFAULT_RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "runs"))
AUTO_DECISIONS = {"ai", "auto"}


def _discover_account_indices(accounts_dir: Path) -> List[int]:
    indices: List[int] = []
    if not accounts_dir.exists():
        return indices

    for entry in accounts_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            value = int(entry.name)
        except (TypeError, ValueError):
            continue
        indices.append(value)

    return sorted(set(indices))


def _load_all_bureaus(
    sid: str, indices: Sequence[int], runs_root: Path
) -> Dict[int, Mapping[str, Mapping[str, Any]]]:
    data: Dict[int, Mapping[str, Mapping[str, Any]]] = {}
    for idx in indices:
        try:
            data[idx] = load_bureaus(sid, idx, runs_root=runs_root)
        except FileNotFoundError:
            data[idx] = {}
        except Exception:
            data[idx] = {}
    return data


def _format_top_parts(parts: Mapping[str, int], limit: int = 5) -> str:
    sortable = [
        (field, int(parts.get(field, 0) or 0))
        for field in _FIELD_SEQUENCE
        if int(parts.get(field, 0) or 0) > 0
    ]
    if not sortable:
        return "-"

    sortable.sort(key=lambda item: (-item[1], item[0]))
    top = [f"{field}={points}" for field, points in sortable[:limit]]
    return ", ".join(top)


def _format_matched_pairs(mapping: Mapping[str, Sequence[str]]) -> str:
    parts: List[str] = []
    for field in _FIELD_SEQUENCE:
        pair = mapping.get(field)
        if not pair:
            continue
        if len(pair) != 2:
            continue
        parts.append(f"{field}={pair[0]}/{pair[1]}")
    return ", ".join(parts) if parts else "-"


def _build_row(
    i: int,
    j: int,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    decision = str(result.get("decision", "different"))
    total = int(result.get("total", 0) or 0)
    mid_sum = int(result.get("mid_sum", 0) or 0)
    dates_all = bool(result.get("dates_all", False))
    triggers = list(result.get("triggers", []))
    conflicts = list(result.get("conflicts", []))
    reasons = list(triggers)
    if conflicts:
        reasons.extend([f"conflict:{name}" for name in conflicts])

    parts = _sanitize_parts(result.get("parts"))
    aux_payload = _build_aux_payload(result.get("aux", {}))
    acctnum_level = aux_payload.get("acctnum_level", "none")
    matched_pairs_map = aux_payload.get("by_field_pairs", {})

    strong_flag = any(str(trigger).startswith("strong:") for trigger in triggers)

    row = {
        "i": int(i),
        "j": int(j),
        "total": total,
        "decision": decision,
        "strong_flag": bool(strong_flag),
        "mid_sum": mid_sum,
        "dates_all": dates_all,
        "acctnum_level": acctnum_level,
        "reasons": reasons,
        "parts": parts,
        "parts_top": _format_top_parts(parts),
        "matched_pairs_map": matched_pairs_map,
        "matched_pairs_display": _format_matched_pairs(matched_pairs_map),
        "result": deepcopy(result),
    }
    return row


def build_pair_rows(
    scores_by_idx: Mapping[int, Mapping[int, Mapping[str, Any]]],
    *,
    only_ai: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in sorted(scores_by_idx.keys()):
        partner_map = scores_by_idx.get(i) or {}
        for j in sorted(partner_map.keys()):
            if j <= i:
                continue
            result = partner_map.get(j)
            if not isinstance(result, Mapping):
                continue
            decision = str(result.get("decision", "different"))
            if only_ai and decision not in AUTO_DECISIONS:
                continue
            rows.append(_build_row(i, j, result))

    rows.sort(key=lambda item: (-item["total"], item["i"], item["j"]))
    return rows


def compute_scores_for_sid(
    sid: str,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    cfg: Optional[Any] = None,
) -> Tuple[List[int], Dict[int, Mapping[int, Mapping[str, Any]]]]:
    accounts_dir = runs_root / sid / "cases" / "accounts"
    indices = _discover_account_indices(accounts_dir)
    if not indices:
        return [], {}

    if cfg is None:
        cfg = get_merge_cfg()

    bureaus = _load_all_bureaus(sid, indices, runs_root)

    scores: Dict[int, Dict[int, Mapping[str, Any]]] = {
        idx: {} for idx in indices
    }

    for pos, left in enumerate(indices):
        left_data = bureaus.get(left, {})
        for right in indices[pos + 1 :]:
            right_data = bureaus.get(right, {})
            result = score_pair(left_data, right_data, cfg)
            scores[left][right] = deepcopy(result)
            scores.setdefault(right, {})[left] = deepcopy(result)

    return indices, scores


def build_merge_tags(
    scores_by_idx: Mapping[int, Mapping[int, Mapping[str, Any]]]
) -> Dict[int, Dict[str, Any]]:
    best_by_idx = choose_best_partner(scores_by_idx)
    tags: Dict[int, Dict[str, Any]] = {}
    for idx in sorted(set(scores_by_idx.keys()) | set(best_by_idx.keys())):
        partner_scores = scores_by_idx.get(idx, {})
        best_info = best_by_idx.get(idx, {})
        tags[idx] = _merge_tag_from_best(idx, partner_scores, best_info)
    return tags


def _print_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    header = (
        "i",
        "j",
        "total",
        "decision",
        "strong",
        "mid",
        "dates",
        "acctnum",
        "reasons",
        "parts",
        "pairs",
    )
    print(" | ".join(header))
    print("-" * 120)
    for row in rows:
        line = " | ".join(
            [
                f"{row['i']}",
                f"{row['j']}",
                f"{row['total']}",
                f"{row['decision']}",
                "Y" if row["strong_flag"] else "N",
                f"{row['mid_sum']}",
                "Y" if row["dates_all"] else "N",
                str(row["acctnum_level"]),
                ", ".join(row["reasons"]) if row["reasons"] else "-",
                row["parts_top"],
                row["matched_pairs_display"],
            ]
        )
        print(line)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True, help="Session identifier")
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT),
        help="Root directory containing run outputs",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write JSON payload mirroring merge tags",
    )
    parser.add_argument(
        "--only-ai",
        action="store_true",
        help="Display only pairs with AI/auto decisions",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display detailed JSON for each pair after the table",
    )

    args = parser.parse_args(argv)

    sid = str(args.sid)
    runs_root = Path(args.runs_root)

    cfg = get_merge_cfg()
    indices, scores_by_idx = compute_scores_for_sid(sid, runs_root=runs_root, cfg=cfg)

    if not indices:
        print(f"No accounts found for SID {sid!r} under {runs_root}")
        return

    rows = build_pair_rows(scores_by_idx, only_ai=bool(args.only_ai))

    if not rows:
        print("No pairs matched the provided filters.")
    else:
        _print_rows(rows)

    merge_tags = build_merge_tags(scores_by_idx)

    if args.show:
        for row in rows:
            print()
            print(f"Pair {row['i']} - {row['j']} result:")
            print(json.dumps(row["result"], indent=2, sort_keys=True))

    if args.json_out:
        output_payload = {
            "sid": sid,
            "pairs": [
                {
                    "i": row["i"],
                    "j": row["j"],
                    "total": row["total"],
                    "decision": row["decision"],
                    "strong_flag": row["strong_flag"],
                    "mid_sum": row["mid_sum"],
                    "dates_all": row["dates_all"],
                    "acctnum_level": row["acctnum_level"],
                    "reasons": row["reasons"],
                    "parts": row["parts"],
                    "matched_pairs_by_field": row["matched_pairs_map"],
                    "result": row["result"],
                }
                for row in rows
            ],
            "merge_tags": merge_tags,
        }
        _write_json(Path(args.json_out), output_payload)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

