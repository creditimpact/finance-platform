#!/usr/bin/env python3
"""Split general information sections from a full token TSV dump.

The original implementation in this repository used a heuristic
``_looks_like_headline`` that treated any all–caps line as a section
heading.  For the credit report ``general info`` tables we only want to
split on a small set of *pre‑defined* headings (``PERSONAL INFORMATION``,
``PUBLIC INFORMATION`` …) and ignore other shouty lines.  This module now
uses an explicit allow‑list which mirrors the logic used by the backend
block segmenter.  Each detected section is exported together with its
start/end coordinates and an incrementing ``section_index`` so consumers
can reference the blocks deterministically.
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
# Utility helpers copied from ``block_segmenter`` for consistency.  These
# helpers mirror the normalisation used throughout the backend so the
# heading matching behaves identically.


def _norm(text: str) -> str:
    """Return an uppercase representation with symbols stripped.

    - Replace NBSP/registration marks
    - Collapse spaces
    - Keep only ``A-Z0-9/&-`` and spaces
    """

    text = (text or "").replace("\u00A0", " ").replace("®", " ")
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[^A-Za-z0-9/&\- ]+", "", text)
    return text.upper()


# Known general-information section headings.  The list intentionally mirrors
# ``SUMMARY_TITLES`` from :mod:`backend.core.logic.report_analysis.block_segmenter`
# so that both components stay in sync if additional headings are supported in
# the future.
SECTION_HEADINGS = {
    "TOTAL ACCOUNTS",
    "CLOSED OR PAID ACCOUNT/ZERO",
    "INQUIRIES",
    "PUBLIC INFORMATION",
    "COLLECTIONS",
    "PERSONAL INFORMATION",
    "SCORE FACTORS",
    "CREDIT SUMMARY",
    "ALERTS",
    "EMPLOYMENT DATA",
}


def _is_anchor(text: str) -> bool:
    """Return ``True`` if ``text`` contains the literal ``Account #`` anchor."""

    return bool(re.search(r"account\s*#", text, re.IGNORECASE))


def _is_section_heading(text: str) -> bool:
    """Return ``True`` if ``text`` matches one of ``SECTION_HEADINGS``."""

    return _norm(text) in SECTION_HEADINGS


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
    index = 1

    for line in lines:
        text = line["text"]
        if _is_anchor(text):
            break
        if _is_section_heading(text):
            if current:
                last = current["lines"][-1]
                current["page_end"] = last["page"]
                current["line_end"] = last["line"]
                sections.append(current)
                index += 1
            current = {
                "section_index": index,
                "heading": text.strip(),
                "page_start": line["page"],
                "line_start": line["line"],
                "lines": [
                    {"page": line["page"], "line": line["line"], "text": text}
                ],
            }
        elif current:
            current["lines"].append(
                {"page": line["page"], "line": line["line"], "text": text}
            )

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
