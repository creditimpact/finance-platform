#!/usr/bin/env python3
"""Split general information sections from a full token TSV dump.

The script groups tokens by ``(page, line)`` to form lines of text.  It scans
for ALL-CAPS headline style lines before the first ``Account #`` anchor and
splits the content into blocks based on those headers.  The resulting blocks
are written to a JSON file for downstream processing or debugging.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Utility helpers copied from ``split_accounts_from_tsv`` for consistency.


def _norm(text: str) -> str:
    """Normalize ``text`` by removing spaces/symbols and lowering case."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_anchor(text: str) -> bool:
    """Return True if ``text`` contains the literal ``Account #`` anchor."""
    return bool(re.search(r"account\s*#", text, re.IGNORECASE))


def _looks_like_headline(text: str) -> bool:
    """Return True if ``text`` is an ALL-CAPS headline candidate."""
    stripped = re.sub(r"[^A-Za-z0-9/&\- ]", "", text).strip()
    if ":" in stripped:
        return False
    core = stripped.replace(" ", "")
    if len(core) < 3:
        return False
    return core.isupper()


# ---------------------------------------------------------------------------
# TSV reading and section splitting


def _read_lines(tsv_path: Path) -> List[Dict[str, Any]]:
    """Read tokens from ``tsv_path`` grouped into consolidated lines."""
    tokens_by_line: Dict[Tuple[int, int], List[Dict[str, str]]] = defaultdict(list)
    with tsv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            page_str = row.get("page")
            line_str = row.get("line")
            if not page_str or not line_str:
                continue
            try:
                page = int(float(page_str))
                line = int(float(line_str))
            except Exception:
                continue
            tokens_by_line[(page, line)].append(row)

    lines: List[Dict[str, Any]] = []
    for page, line in sorted(tokens_by_line.keys()):
        text = "".join(tok.get("text", "") for tok in tokens_by_line[(page, line)])
        lines.append({"page": page, "line": line, "text": text})
    return lines


def split_general_info(tsv_path: Path, json_out: Path) -> Dict[str, Any]:
    """Split general information blocks from ``tsv_path`` and write JSON."""
    lines = _read_lines(tsv_path)
    sections: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    for line in lines:
        text = line["text"]
        if _is_anchor(text):
            break
        if _looks_like_headline(text):
            if current:
                last = current["lines"][-1]
                current["page_end"] = last["page"]
                current["line_end"] = last["line"]
                sections.append(current)
            current = {
                "heading": text.strip(),
                "page_start": line["page"],
                "line_start": line["line"],
                "lines": [line],
            }
        elif current:
            current["lines"].append(line)

    if current and current.get("lines"):
        last = current["lines"][-1]
        current["page_end"] = last["page"]
        current["line_end"] = last["line"]
        sections.append(current)

    result = {"sections": sections}
    json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# CLI entry point


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Split general information sections from the full TSV"
    )
    ap.add_argument("--full", default="_debug_full.tsv", help="Input TSV path")
    ap.add_argument(
        "--json_out", default="general_info_from_full.json", help="JSON output path"
    )
    args = ap.parse_args(argv)

    tsv_path = Path(args.full)
    json_out = Path(args.json_out)
    split_general_info(tsv_path, json_out)
    print(f"Wrote general info sections to {json_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
