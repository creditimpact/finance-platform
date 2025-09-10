from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.core.logic.report_analysis.trace_cleanup import purge_trace_except_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely purge trace files except final artifacts")
    parser.add_argument("--sid", required=True, help="Session identifier")
    parser.add_argument("--root", default=".", help="Project root path")
    parser.add_argument("--dry-run", action="store_true", help="Preview deletions without removing files")
    parser.add_argument(
        "--keep-extra",
        action="append",
        default=[],
        help="Additional relative paths under the session to keep",
    )
    args = parser.parse_args()
    summary = purge_trace_except_artifacts(
        sid=args.sid,
        root=Path(args.root),
        keep_extra=args.keep_extra or None,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
