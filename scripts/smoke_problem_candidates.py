import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Optional bootstrap to ensure repo root is on sys.path
try:  # pragma: no cover - convenience import
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:
    # Fallback: add repository root based on this file's location
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from collections import Counter
from backend.core.logic.report_analysis.account_merge import (
    DEFAULT_CFG,
    cluster_problematic_accounts,
)
from backend.core.logic.report_analysis.problem_extractor import detect_problem_accounts


def _build_merge_summary(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return clusters and pair table summaries from merge-tagged candidates."""

    cluster_map: Dict[str, Dict[str, Any]] = {}
    pairs: List[Dict[str, Any]] = []

    for idx, account in enumerate(candidates):
        merge_tag = account.get("merge_tag") or {}
        group_id = str(merge_tag.get("group_id") or f"g{idx + 1}")
        best_match = merge_tag.get("best_match") or {}
        parts = merge_tag.get("parts") or {}

        entry: Dict[str, Any] = {
            "idx": idx,
            "group": group_id,
            "best": best_match.get("account_index"),
            "score": best_match.get("score"),
            "decision": merge_tag.get("decision"),
        }

        pair_decision = best_match.get("decision")
        if pair_decision is not None and pair_decision != entry.get("decision"):
            entry["pair_decision"] = pair_decision

        if parts:
            entry["parts"] = {key: float(value) for key, value in parts.items()}

        pairs.append(entry)

        cluster_entry = cluster_map.setdefault(
            group_id,
            {"group": group_id, "members": [], "best_matches": []},
        )
        cluster_entry["members"].append(idx)
        cluster_entry["best_matches"].append(entry)

    pairs.sort(key=lambda item: item["idx"])

    clusters: List[Dict[str, Any]] = []
    for group_id in sorted(cluster_map.keys()):
        cluster_entry = cluster_map[group_id]
        cluster_entry["members"].sort()
        cluster_entry["best_matches"] = sorted(
            cluster_entry["best_matches"], key=lambda item: item["idx"]
        )
        clusters.append(cluster_entry)

    return {"clusters": clusters, "pairs": pairs}


def check_lean(sid: str) -> None:
    """Validate that case folders for the SID contain lean artifacts only."""

    base = Path("runs") / sid / "cases" / "accounts"
    if not base.exists():
        raise FileNotFoundError(f"accounts directory not found for SID {sid!r} at {base}")

    required_files = (
        "meta.json",
        "summary.json",
        "bureaus.json",
        "fields_flat.json",
        "raw_lines.json",
        "tags.json",
    )

    for account_dir in base.iterdir():
        if not account_dir.is_dir():
            continue

        for filename in required_files:
            path = account_dir / filename
            if not path.exists():
                raise AssertionError(
                    f"expected {filename} in {account_dir}, but it was missing"
                )

        summary_text = (account_dir / "summary.json").read_text(encoding="utf-8")
        if "triad_rows" in summary_text:
            raise AssertionError(
                f"triad_rows detected in summary.json for {account_dir}; expected lean output"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke run problem candidate detection for a run SID")
    ap.add_argument("--sid", required=True, help="Run session id (SID) to analyze")
    ap.add_argument(
        "--show-all",
        action="store_true",
        help="Show all candidates instead of first 5",
    )
    ap.add_argument(
        "--show-merge",
        action="store_true",
        help="Display merge clustering summary",
    )
    ap.add_argument(
        "--check-lean",
        action="store_true",
        help="Verify case folders contain lean artifacts with no triad_rows",
    )
    ap.add_argument("--json-out", help="Path to write full JSON payload including merge data")
    args = ap.parse_args()

    out = detect_problem_accounts(args.sid)

    # Primary issue frequency
    issue_counts = Counter([str(c.get("reason", {}).get("primary_issue")) for c in out])
    issues_line = ", ".join(f"{k}={v}" for k, v in issue_counts.items() if k and v)

    # Reason key frequency (prefix before ':')
    reason_keys = []
    for c in out:
        reasons = (c.get("reason", {}) or {}).get("problem_reasons") or []
        for r in reasons:
            s = str(r)
            key = s.split(":", 1)[0].strip() if ":" in s else s.strip()
            if key:
                reason_keys.append(key)
    reason_counts = Counter(reason_keys)
    top_reasons = ", ".join(f"{k}={v}" for k, v in reason_counts.most_common(10))

    merge_summary: Dict[str, Any] | None = None
    if args.show_merge:
        out = cluster_problematic_accounts(out, DEFAULT_CFG, sid=args.sid)
        merge_summary = _build_merge_summary(out)

    payload = {
        "sid": args.sid,
        "problematic": len(out),
        "sample": out if args.show_all else out[:5],
    }

    if merge_summary is not None:
        payload["merge"] = merge_summary

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if issues_line:
        print(f"issues: {issues_line}")
    if top_reasons:
        print(f"reasons: {top_reasons}")

    if merge_summary is not None:
        clusters_view: List[Dict[str, Any]] = []
        for cluster in merge_summary["clusters"]:
            clusters_view.append(
                {
                    "group": cluster["group"],
                    "members": cluster["members"],
                    "best_matches": [
                        {
                            key: entry[key]
                            for key in ("idx", "best", "score", "decision")
                            if key in entry
                        }
                        for entry in cluster["best_matches"]
                    ],
                }
            )

        pairs_view = [
            {key: entry[key] for key in ("idx", "best", "score", "decision") if key in entry}
            for entry in merge_summary["pairs"]
        ]

        print("clusters:", json.dumps(clusters_view, ensure_ascii=False, indent=2))
        print("pairs:", json.dumps(pairs_view, ensure_ascii=False, indent=2))

    json_out_path = Path(args.json_out) if args.json_out else None
    if json_out_path is not None:
        json_payload: Dict[str, Any] = {
            "sid": args.sid,
            "problematic": len(out),
            "candidates": out,
            "issue_counts": dict(issue_counts),
            "reason_counts": dict(reason_counts),
        }
        if merge_summary is not None:
            json_payload["merge"] = merge_summary

        json_out_path.write_text(
            json.dumps(json_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote JSON to {json_out_path}")

    if args.check_lean:
        check_lean(args.sid)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
