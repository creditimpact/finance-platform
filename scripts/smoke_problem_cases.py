from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from backend.core.logic.report_analysis.problem_extractor import detect_problem_accounts
from backend.core.logic.report_analysis.problem_case_builder import build_problem_cases


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Smoke test helper for problem account extraction"
    )
    ap.add_argument("--sid", required=True, help="Session ID under traces/blocks")
    ap.add_argument(
        "--root", default=None, help="Optional repository root (defaults to backend.settings.PROJECT_ROOT)"
    )
    args = ap.parse_args()

    sid = args.sid
    root = Path(args.root) if args.root else None

    candidates: List[Dict[str, Any]] = detect_problem_accounts(sid, root=root)
    summary = build_problem_cases(sid, candidates, root=root)

    payload = {"sid": sid, "found": candidates, "summary": summary}
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
