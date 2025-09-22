"""Build AI adjudication packs for merge V2."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - convenience bootstrap
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback for direct execution
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from backend.core.logic.report_analysis.ai_packs import build_merge_ai_packs
from backend.pipeline.runs import RunManifest, persist_manifest


log = logging.getLogger(__name__)


def _packs_dir_for(sid: str, runs_root: Path) -> Path:
    """Return the canonical ``ai_packs`` directory for ``sid``."""

    from backend.pipeline.auto_ai import packs_dir_for as _packs_dest  # local import to avoid cycles

    return _packs_dest(sid, runs_root=runs_root)


def _write_pack(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def _write_index(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(serialized + "\n", encoding="utf-8")


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sid", required=True, help="Session identifier")
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory containing runs/<SID> outputs",
    )
    parser.add_argument(
        "--packs-dir",
        help="Destination directory for generated AI packs (defaults to runs/<SID>/ai_packs)",
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
    packs_dir = Path(args.packs_dir) if args.packs_dir else _packs_dir_for(sid, runs_root)
    packs_dir.mkdir(parents=True, exist_ok=True)
    log.info("PACKS_DIR_USED sid=%s dir=%s", sid, packs_dir)

    packs = build_merge_ai_packs(
        sid,
        runs_root,
        only_merge_best=bool(args.only_merge_best),
        max_lines_per_side=int(args.max_lines_per_side),
    )

    index_entries: list[dict[str, object]] = []
    for pack in packs:
        pair = pack.get("pair") or {}
        try:
            a_idx = int(pair.get("a"))
            b_idx = int(pair.get("b"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Pack is missing pair indices") from exc

        pack_filename = f"pair_{a_idx:03d}_{b_idx:03d}.jsonl"
        pack_path = packs_dir / pack_filename
        context = pack.get("context") if isinstance(pack.get("context"), dict) else {}
        context_a = context.get("a") if isinstance(context.get("a"), list) else []
        context_b = context.get("b") if isinstance(context.get("b"), list) else []
        highlights = pack.get("highlights") if isinstance(pack.get("highlights"), dict) else {}
        score_total = _safe_int(highlights.get("total"))

        _write_pack(pack_path, pack)
        log.info("PACK_WRITTEN sid=%s file=%s", sid, pack_filename)

        index_entries.append(
            {
                "a": a_idx,
                "b": b_idx,
                "pack_file": pack_filename,
                "lines_a": len(context_a),
                "lines_b": len(context_b),
                "score_total": score_total,
            }
        )

    pairs_count = len(index_entries)
    if pairs_count > 0:
        index_path = packs_dir / "index.json"
        index_payload = {
            "sid": sid,
            "packs": index_entries,
            "pairs_count": pairs_count,
        }
        _write_index(index_path, index_payload)
        log.info(
            "INDEX_WRITTEN sid=%s index=%s pairs=%s",
            sid,
            index_path,
            len(index_payload.get("packs", [])),
        )

        manifest = RunManifest.for_sid(sid)
        manifest.set_ai_built(packs_dir, len(index_payload.get("packs", [])))
        persist_manifest(manifest)
        log.info(
            "MANIFEST_AI_PACKS_UPDATED sid=%s dir=%s pairs=%s",
            sid,
            packs_dir,
            len(index_payload.get("packs", [])),
        )
    else:
        log.info("INDEX_SKIPPED_NO_PAIRS sid=%s dir=%s", sid, packs_dir)

    print(f"[BUILD] wrote {len(index_entries)} packs to {packs_dir}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
