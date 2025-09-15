try:  # pragma: no cover - import shim
    import scripts._bootstrap  # KEEP FIRST
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parent.parent
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    import scripts._bootstrap  # type: ignore  # KEEP FIRST

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from backend.core.logic.report_analysis.problem_case_builder import build_problem_cases
from backend.core.logic.report_analysis.problem_extractor import detect_problem_accounts
from backend.pipeline.runs import RunManifest


def _resolve_manifest(manifest_arg: str | None) -> RunManifest:
    if manifest_arg:
        return RunManifest(Path(manifest_arg)).load()
    return RunManifest.from_env_or_latest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Smoke test helper for problem account extraction",
    )
    ap.add_argument("--manifest", help="Path to run manifest")
    args = ap.parse_args()

    m = _resolve_manifest(args.manifest)
    accounts_path = Path(m.get("traces.accounts_table", "accounts_json"))
    general_path = Path(m.get("traces.accounts_table", "general_json"))
    root = accounts_path.parent.parent.parent.parent
    sid = m.sid

    candidates: List[Dict[str, Any]] = detect_problem_accounts(sid, root=root)
    summary = build_problem_cases(sid, candidates, root=root)

    payload = {"sid": sid, "found": candidates, "summary": summary}
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

