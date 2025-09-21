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
from backend.pipeline.runs import RunManifest, persist_manifest


def _write_json_file(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(serialized + "\n", encoding="utf-8")


def _update_run_manifest_stub(
    run_dir: Path,
    sid: str,
    *,
    ai_packs: Mapping[str, Path | str],
) -> None:
    """Update ``runs/<SID>/.manifest`` with the AI pack artifact metadata.

    The ``.manifest`` file serves as a lightweight manifest for tooling that
    consumes the run outputs directly (e.g., Codex automations).  Historical
    runs may have stored only a plain-text pointer in this location.  For
    compatibility, retain that information while merging in the structured
    ``artifacts.ai_packs`` payload required by the automation stack.
    """

    manifest_path = run_dir / ".manifest"
    if manifest_path.exists():
        raw = manifest_path.read_text(encoding="utf-8").strip()
        if raw:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"manifest_path": raw}
        else:
            data = {}
    else:
        data = {}

    if not isinstance(data, dict):
        data = {}

    data.setdefault("sid", sid)

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        data["artifacts"] = artifacts

    artifacts["ai_packs"] = {
        key: str(Path(value).resolve()) for key, value in ai_packs.items()
    }

    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    manifest_path.write_text(serialized + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True, help="Session identifier")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory containing runs/<SID> outputs",
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
    out_dir = runs_root / sid / "ai_packs"

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
        _write_json_file(path, pack)
        index.append({"a": a_idx, "b": b_idx, "file": filename})

    index_path = out_dir / "index.json"
    _write_json_file(index_path, index)

    logs_path = out_dir / "logs.txt"
    logs_path.touch(exist_ok=True)

    manifest = RunManifest.for_sid(sid)
    persist_manifest(
        manifest,
        artifacts={
            "ai_packs": {
                "dir": out_dir,
                "index": index_path,
                "logs": logs_path,
            }
        },
    )

    _update_run_manifest_stub(
        runs_root / sid,
        sid,
        ai_packs={
            "dir": out_dir,
            "index": index_path,
            "logs": logs_path,
        },
    )

    print(f"[BUILD] wrote {len(packs)} packs to {out_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

