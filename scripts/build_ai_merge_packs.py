"""Build AI adjudication packs for merge V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

try:  # pragma: no cover - convenience bootstrap
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback for direct execution
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from backend.core.logic.report_analysis.ai_packs import build_merge_ai_packs
from backend.pipeline.runs import RunManifest


def _write_pack(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(serialized + "\n", encoding="utf-8")


def _resolve_out_dir(base: Path, sid: str, out_dir_arg: str | None) -> Path:
    if out_dir_arg:
        return Path(out_dir_arg)
    return base / sid / "ai_packs"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True, help="Session identifier")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory containing runs/<SID> outputs",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory for packs (defaults to runs/<SID>/ai_packs)",
    )
    parser.add_argument(
        "--max-lines-per-side",
        type=int,
        default=20,
        help="Maximum number of context lines per account",
    )
    parser.add_argument(
        "--only-merge-best",
        dest="only_merge_best",
        action="store_true",
        help="Include only pairs marked as merge_best (default)",
    )
    parser.add_argument(
        "--include-all-pairs",
        dest="only_merge_best",
        action="store_false",
        help="Include all AI pairs regardless of merge_best",
    )
    parser.set_defaults(only_merge_best=True)

    args = parser.parse_args(argv)

    sid = str(args.sid)
    runs_root = Path(args.runs_root)
    out_dir = _resolve_out_dir(runs_root, sid, args.out_dir)

    packs = build_merge_ai_packs(
        sid,
        runs_root,
        only_merge_best=bool(args.only_merge_best),
        max_lines_per_side=int(args.max_lines_per_side),
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict[str, object]] = []
    for pack in packs:
        pair = pack.get("pair") or {}
        try:
            a_idx = int(pair.get("a"))
            b_idx = int(pair.get("b"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Pack is missing pair indices") from exc
        filename = f"{a_idx:03d}-{b_idx:03d}.json"
        path = out_dir / filename
        _write_pack(path, pack)
        index.append({"a": a_idx, "b": b_idx, "file": str(path.resolve())})

    manifest = RunManifest.for_sid(sid)
    manifest.set_artifact("ai", "packs_dir", out_dir.resolve())
    ai_group = manifest.data.setdefault("artifacts", {}).setdefault("ai", {})
    ai_group["pairs"] = index
    manifest.save()

    print(f"[BUILD] wrote {len(packs)} packs to {out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

