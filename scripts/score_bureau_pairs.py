"""CLI for inspecting deterministic account merge scores between bureau accounts."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from backend.core.logic.report_analysis.account_merge import score_all_pairs_0_100


DEFAULT_RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "runs"))
AUTO_DECISIONS = {"ai", "auto"}

FIELD_SEQUENCE: Tuple[str, ...] = (
    "balance_owed",
    "account_number",
    "last_payment",
    "past_due_amount",
    "high_balance",
    "creditor_type",
    "account_type",
    "payment_amount",
    "credit_limit",
    "last_verified",
    "date_of_last_activity",
    "date_reported",
    "date_opened",
    "closed_date",
)


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


def _format_top_parts(parts: Mapping[str, int], limit: int = 5) -> str:
    sortable = []
    for field in FIELD_SEQUENCE:
        try:
            points = int(parts.get(field, 0) or 0)
        except (TypeError, ValueError):
            points = 0
        if points > 0:
            sortable.append((field, points))
    if not sortable:
        return "-"

    sortable.sort(key=lambda item: (-item[1], item[0]))
    top = [f"{field}={points}" for field, points in sortable[:limit]]
    return ", ".join(top)


def _format_matched_pairs(mapping: Mapping[str, Sequence[str]]) -> str:
    parts: List[str] = []
    for field in FIELD_SEQUENCE:
        pair = mapping.get(field)
        if not pair:
            continue
        if len(pair) != 2:
            continue
        parts.append(f"{field}={pair[0]}/{pair[1]}")
    return ", ".join(parts) if parts else "-"


def _sanitize_parts(parts: Optional[Mapping[str, Any]]) -> Dict[str, int]:
    sanitized: Dict[str, int] = {}
    for field in FIELD_SEQUENCE:
        value = 0
        if isinstance(parts, Mapping):
            try:
                value = int(parts.get(field, 0) or 0)
            except (TypeError, ValueError):
                value = 0
        sanitized[field] = value
    return sanitized


def _extract_aux_payload(aux: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    acct_level = "none"
    by_field_pairs: Dict[str, List[str]] = {}

    if isinstance(aux, Mapping):
        acct_aux = aux.get("account_number")
        if isinstance(acct_aux, Mapping):
            level = acct_aux.get("acctnum_level")
            if isinstance(level, str) and level:
                acct_level = level

        for field in FIELD_SEQUENCE:
            field_aux = aux.get(field)
            if not isinstance(field_aux, Mapping):
                continue
            best_pair = field_aux.get("best_pair")
            if (
                isinstance(best_pair, Iterable)
                and not isinstance(best_pair, (str, bytes))
            ):
                pair_list = list(best_pair)
                if len(pair_list) == 2:
                    by_field_pairs[field] = [str(pair_list[0]), str(pair_list[1])]

    return {"acctnum_level": acct_level, "by_field_pairs": by_field_pairs}


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
    aux_payload = _extract_aux_payload(result.get("aux", {}))
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
) -> Tuple[List[int], Dict[int, Mapping[int, Mapping[str, Any]]]]:
    accounts_dir = runs_root / sid / "cases" / "accounts"
    indices = _discover_account_indices(accounts_dir)
    if not indices:
        return [], {}

    scores = score_all_pairs_0_100(sid, indices, runs_root=runs_root)

    for idx in indices:
        scores.setdefault(idx, {})

    return indices, scores


@lru_cache(maxsize=1)
def _get_account_merge_module():
    return import_module("backend.core.logic.report_analysis.account_merge")


def choose_best_partner_cached(
    scores_by_idx: Mapping[int, Mapping[int, Mapping[str, Any]]]
) -> Dict[int, Dict[str, Any]]:
    module = _get_account_merge_module()
    return module.choose_best_partner(scores_by_idx)


def build_merge_tags(
    scores_by_idx: Mapping[int, Mapping[int, Mapping[str, Any]]],
    best_by_idx: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Dict[int, Dict[str, Any]]:
    if best_by_idx is None:
        best_by_idx = choose_best_partner_cached(scores_by_idx)
    module = _get_account_merge_module()
    tags: Dict[int, Dict[str, Any]] = {}
    for idx in sorted(set(scores_by_idx.keys()) | set(best_by_idx.keys())):
        partner_scores = scores_by_idx.get(idx, {})
        best_info = best_by_idx.get(idx, {})
        tags[idx] = module._merge_tag_from_best(idx, partner_scores, best_info)
    return tags


def persist_merge_tags_to_tags(
    sid: str,
    scores_by_idx: Mapping[int, Mapping[int, Mapping[str, Any]]],
    best_by_idx: Mapping[int, Mapping[str, Any]],
    *,
    runs_root: Path,
) -> Dict[int, Dict[str, Any]]:
    module = _get_account_merge_module()
    return module.persist_merge_tags(
        sid, scores_by_idx, best_by_idx, runs_root=runs_root
    )


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
    parser.add_argument(
        "--write-tags",
        action="store_true",
        help="Persist merge tags to per-account tags.json files",
    )

    args = parser.parse_args(argv)

    sid = str(args.sid)
    runs_root = Path(args.runs_root)

    indices, scores_by_idx = compute_scores_for_sid(sid, runs_root=runs_root)

    if not indices:
        print(f"No accounts found for SID {sid!r} under {runs_root}")
        return

    best_by_idx = choose_best_partner_cached(scores_by_idx)
    rows = build_pair_rows(scores_by_idx, only_ai=bool(args.only_ai))

    if not rows:
        print("No pairs matched the provided filters.")
    else:
        _print_rows(rows)

    merge_tags = build_merge_tags(scores_by_idx, best_by_idx)

    if args.write_tags:
        merge_tags = persist_merge_tags_to_tags(
            sid, scores_by_idx, best_by_idx, runs_root=runs_root
        )

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

