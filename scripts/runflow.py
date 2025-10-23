try:  # pragma: no cover - import shim to support direct execution
    import scripts._bootstrap  # KEEP FIRST
except ModuleNotFoundError:  # pragma: no cover - fallback path setup
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parent.parent
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    import scripts._bootstrap  # type: ignore  # KEEP FIRST

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from backend.ai.validation_index import ValidationPackIndexWriter
from backend.validation.index_schema import ValidationIndex, ValidationPackRecord, load_validation_index

from backend.runflow.decider import (
    refresh_validation_stage_from_index,
    reconcile_umbrella_barriers,
    runflow_refresh_umbrella_barriers,
    _validation_record_has_results,
    _validation_record_result_paths,
)


def _resolve_runs_root(value: str | None) -> Path:
    if not value:
        return Path("runs").resolve()
    return Path(value).resolve()


def _load_validation_index(index_path: Path) -> ValidationIndex:
    try:
        return load_validation_index(index_path)
    except FileNotFoundError:
        raise SystemExit(f"validation index not found: {index_path}")


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError:
        return 0
    except OSError:
        return 0


def _select_result_path(paths: Sequence[Path]) -> Path | None:
    if not paths:
        return None
    for candidate in paths:
        if candidate.suffix.lower() == ".json":
            return candidate
    return paths[0]


def _record_completed_result(
    writer: ValidationPackIndexWriter,
    index: ValidationIndex,
    record: ValidationPackRecord,
    *,
    result_paths: Sequence[Path],
) -> bool:
    summary_path = _select_result_path(result_paths)
    if summary_path is None:
        return False

    pack_path = index.resolve_pack_path(record)
    request_lines = None
    model = None
    completed_at = None

    extra = getattr(record, "extra", None)
    if isinstance(extra, Mapping):
        request_lines = extra.get("request_lines")
        model = extra.get("model") or extra.get("ai_model")
        completed_at = extra.get("completed_at")

    line_count = record.lines if getattr(record, "lines", 0) > 0 else None
    if line_count is None and summary_path.suffix.lower() == ".jsonl":
        line_count = _count_lines(summary_path)

    updated = writer.record_result(
        pack_path,
        status="completed",
        error=None,
        request_lines=request_lines,
        model=model,
        completed_at=completed_at,
        result_path=summary_path,
        line_count=line_count,
    )

    return updated is not None


def _cmd_backfill_validation(args: argparse.Namespace) -> int:
    sid = (args.sid or "").strip()
    if not sid:
        print("error: SID is required", file=sys.stderr)
        return 2

    runs_root = _resolve_runs_root(getattr(args, "runs_root", None))
    run_dir = runs_root / sid
    index_path = run_dir / "ai_packs" / "validation" / "index.json"

    index = _load_validation_index(index_path)
    writer = ValidationPackIndexWriter(
        sid=sid,
        index_path=index_path,
        packs_dir=index.packs_dir_path,
        results_dir=index.results_dir_path,
    )

    completed = 0
    missing_results: list[int] = []

    for record in index.packs:
        normalized_status = (record.status or "").strip().lower()
        has_results = _validation_record_has_results(index, record)
        if not has_results:
            if normalized_status == "completed":
                missing_results.append(record.account_id)
            continue

        if normalized_status == "completed":
            continue

        result_paths = _validation_record_result_paths(index, record)
        if not result_paths:
            continue

        if _record_completed_result(writer, index, record, result_paths=result_paths):
            completed += 1

    refresh_validation_stage_from_index(sid, runs_root=runs_root)
    runflow_refresh_umbrella_barriers(sid)
    reconcile_umbrella_barriers(sid, runs_root=runs_root)

    payload = {
        "sid": sid,
        "updated": completed,
        "missing_results": missing_results,
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runflow management commands")
    sub = parser.add_subparsers(dest="command", required=True)

    backfill = sub.add_parser(
        "backfill-validation",
        help="Backfill validation index entries based on on-disk results",
    )
    backfill.add_argument("sid", help="Run identifier")
    backfill.add_argument(
        "--runs-root",
        dest="runs_root",
        help="Override the runs root directory",
    )
    backfill.set_defaults(func=_cmd_backfill_validation)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.error("a subcommand is required")
    return handler(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
