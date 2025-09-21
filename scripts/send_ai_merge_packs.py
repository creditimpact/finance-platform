"""Send merge V2 AI packs to the adjudicator service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

try:  # pragma: no cover - convenience bootstrap for direct execution
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback when bootstrap is unavailable
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from backend.core.logic.report_analysis import ai_sender
from backend.pipeline.runs import RunManifest, persist_manifest


def _load_index(path: Path) -> list[Mapping[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Pack index must be a list: {path}")
    return [dict(entry) for entry in data]


def _load_pack(path: Path) -> Mapping[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"AI pack must be a JSON object: {path}")
    return data


def _resolve_packs_dir(runs_root: Path, sid: str, override: str | None) -> Path:
    if override:
        return Path(override)
    return runs_root / sid / "ai_packs"


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _log_factory(path: Path, sid: str, pair: Mapping[str, int], file_name: str):
    def _log(event: str, payload: Mapping[str, object] | None = None) -> None:
        extras: dict[str, object] = {
            "sid": sid,
            "pair": {"a": pair["a"], "b": pair["b"]},
            "file": file_name,
        }
        if payload:
            extras.update(payload)
        serialized = json.dumps(extras, ensure_ascii=False, sort_keys=True)
        line = f"{ai_sender.isoformat_timestamp()} AI_ADJUDICATOR_{event} {serialized}\n"
        _append_log(path, line)

    return _log


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
        default=None,
        help="Optional override for the directory containing ai packs",
    )
    args = parser.parse_args(argv)

    if not ai_sender.is_enabled():
        print("[AI] adjudicator disabled; skipping")
        return

    config = ai_sender.load_config_from_env()

    sid = str(args.sid)
    runs_root = Path(args.runs_root)
    packs_dir = _resolve_packs_dir(runs_root, sid, args.packs_dir)
    index_path = packs_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Pack index not found: {index_path}")

    index = _load_index(index_path)
    logs_path = packs_dir / "logs.txt"

    manifest = RunManifest.for_sid(sid)
    persist_manifest(
        manifest,
        artifacts={
            "ai_packs": {
                "dir": packs_dir,
                "index": index_path,
                "logs": logs_path,
            }
        },
    )

    total = 0
    successes = 0
    failures = 0

    for entry in index:
        if "a" not in entry or "b" not in entry or "file" not in entry:
            raise ValueError(f"Invalid pack index entry: {entry}")
        a_idx = int(entry["a"])
        b_idx = int(entry["b"])
        pack_path = packs_dir / str(entry["file"])
        if not pack_path.exists():
            raise FileNotFoundError(f"Pack file missing: {pack_path}")

        pack = _load_pack(pack_path)
        total += 1

        log = _log_factory(logs_path, sid, {"a": a_idx, "b": b_idx}, pack_path.name)
        log("PACK_START", {})

        outcome = ai_sender.process_pack(pack, config, log=log)
        timestamp = ai_sender.isoformat_timestamp()

        if outcome.success and outcome.decision and outcome.reason:
            ai_sender.write_decision_tags(
                runs_root,
                sid,
                a_idx,
                b_idx,
                outcome.decision,
                outcome.reason,
                timestamp,
            )
            log(
                "PACK_SUCCESS",
                {"decision": outcome.decision, "reason": outcome.reason},
            )
            successes += 1
        else:
            ai_sender.write_error_tags(
                runs_root,
                sid,
                a_idx,
                b_idx,
                outcome.error_kind or "Error",
                outcome.error_message or "",
                timestamp,
            )
            log(
                "PACK_FAILURE",
                {
                    "error_kind": outcome.error_kind or "Error",
                },
            )
            failures += 1

    print(
        "[AI] adjudicated {total} packs ({successes} success, {failures} errors)".format(
            total=total, successes=successes, failures=failures
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

