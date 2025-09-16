import argparse
import json
import sys
from pathlib import Path

# Optional bootstrap to ensure repo root is on sys.path
try:  # pragma: no cover - convenience import
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:
    # Fallback: add repository root based on this file's location
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from collections import Counter
from backend.core.logic.report_analysis.problem_extractor import detect_problem_accounts


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke run problem candidate detection for a run SID")
    ap.add_argument("--sid", required=True, help="Run session id (SID) to analyze")
    ap.add_argument("--show-all", action="store_true", help="Show all candidates instead of first 5")
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

    payload = {
        "sid": args.sid,
        "problematic": len(out),
        "sample": out if args.show_all else out[:5],
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if issues_line:
        print(f"issues: {issues_line}")
    if top_reasons:
        print(f"reasons: {top_reasons}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
