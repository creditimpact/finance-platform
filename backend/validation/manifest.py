"""Command-line helpers for working with validation manifest indexes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

from backend.core.ai.paths import validation_index_path

from .index_schema import ValidationIndex, load_validation_index


def load_index_for_sid(sid: str, *, runs_root: Path | str | None = None) -> ValidationIndex:
    """Return the :class:`ValidationIndex` for ``sid``."""

    index_path = validation_index_path(sid, runs_root=runs_root, create=False)
    return load_validation_index(index_path)


def check_index(
    index: ValidationIndex,
    *,
    stream: TextIO = sys.stdout,
) -> bool:
    """Verify that every pack referenced by ``index`` exists on disk."""

    ok = True
    for record in index.packs:
        pack_path = index.resolve_pack_path(record)
        status = "OK" if pack_path.is_file() else "MISSING"
        stream.write(
            f"{status:>8}  account {record.account_id:03d}  {record.pack}\n"
        )
        if status == "MISSING":
            ok = False

    if ok:
        stream.write(
            f"All {len(index.packs)} packs present for SID {index.sid}.\n"
        )
    else:
        stream.write("Missing packs detected.\n")
    return ok


def _parse_argv(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation manifest utilities")
    parser.add_argument("--sid", required=True, help="Run SID to inspect")
    parser.add_argument(
        "--runs-root",
        help="Base runs/ directory (defaults to ./runs or RUNS_ROOT env)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that every pack referenced by the index exists",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_argv(argv)
    runs_root = Path(args.runs_root) if args.runs_root else None

    try:
        index = load_index_for_sid(args.sid, runs_root=runs_root)
    except FileNotFoundError:
        print(
            f"Validation index not found for SID {args.sid!r}.",
            file=sys.stderr,
        )
        return 2
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(
            f"Unable to load validation index for SID {args.sid!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    if args.check:
        ok = check_index(index)
        return 0 if ok else 1

    print(index.index_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

