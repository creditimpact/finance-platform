#!/usr/bin/env python3
"""Split general information sections from a full token TSV dump.

This helper is used when working with raw token TSV dumps of full credit
reports.  The general‑information area at the start of the report is made up
of a handful of well known sections ("Personal Information", "Summary", ...).
Rather than relying on heuristics we use explicit start/end boundaries for
those headings so the resulting JSON is deterministic and mirrors the logic
used by the backend segmenter.
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


def _is_anchor(text: str) -> bool:
    """Return ``True`` if ``text`` contains the literal ``Account #`` anchor."""

    return bool(re.search(r"account\s*#", text, re.IGNORECASE))


# Ordered list of section boundaries.  Each tuple contains
# ``(start_heading, end_heading, include_end)``.  ``include_end`` controls
# whether the line containing ``end_heading`` should be part of the section.
SECTION_RULES: List[Tuple[str, str, bool]] = [
    ("PERSONAL INFORMATION", "SUMMARY", False),
    ("SUMMARY", "ACCOUNT HISTORY", False),
    ("ACCOUNT HISTORY", "COLLECTION CHARGEOFF", True),
    ("PUBLIC INFORMATION", "INQUIRIES", False),
    ("INQUIRIES", "CREDITOR CONTACTS", False),
    ("CREDITOR CONTACTS", "SMARTCREDIT", False),
]


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

    # Stop processing once the account anchor is reached; the general info
    # section always precedes the accounts table.
    for idx, line in enumerate(lines):
        if _is_anchor(line["text"]):
            lines = lines[:idx]
            break

    norm_lines = [_norm(ln["text"]) for ln in lines]

    sections: List[Dict[str, Any]] = []
    index = 1
    for start, end, include_end in SECTION_RULES:
        start_norm = _norm(start)
        end_norm = _norm(end)
        try:
            start_idx = norm_lines.index(start_norm)
            end_idx = norm_lines.index(end_norm, start_idx + 1)
        except ValueError:
            # Missing start or end heading; skip this section silently.
            continue

        slice_end = end_idx + 1 if include_end else end_idx
        section_lines = lines[start_idx:slice_end]
        if not section_lines:
            continue
        last = section_lines[-1]
        sections.append(
            {
                "section_index": index,
                "heading": section_lines[0]["text"].strip(),
                "page_start": section_lines[0]["page"],
                "line_start": section_lines[0]["line"],
                "page_end": last["page"],
                "line_end": last["line"],
                "lines": [
                    {
                        "page": ln["page"],
                        "line": ln["line"],
                        "text": ln["text"],
                    }
                    for ln in section_lines
                ],
            }
        )
        index += 1

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
