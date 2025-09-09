#!/usr/bin/env python3
"""Split accounts from a full token TSV dump.

This script groups tokens by `(page, line)` to form lines of text. It detects
account boundaries based on lines that contain the phrase ``Account`` followed
by a ``#`` on the same line. Each account is emitted to a structured JSON file
and, optionally, into individual TSV files for debugging.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from bisect import bisect_right

ACCOUNT_RE = re.compile(r"\bAccount\b.*#", re.IGNORECASE)
STOP_MARKER_NORM = "publicinformation"
SECTION_STARTERS = {"collection"}
_SECTION_NAME = {"collection": "collections"}

# How many lines above the ``Account #`` anchor to consider the heading.
# If unavailable, the logic falls back to progressively closer lines.
HEADING_BACK_LINES = 2


def _norm(text: str) -> str:
    """Normalize ``text`` by removing spaces/symbols and lowering case."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


BUREAUS_NORM = _norm("Transunion®Experian®Equifax®")


def _is_heading(text: str) -> bool:
    """Return True if ``text`` appears to be an ALL-CAPS heading."""
    candidate = text.strip()
    if not candidate:
        return False
    # ``str.isupper`` handles alphanumeric with punctuation reasonably well but
    # fails on strings without alphabetic characters. Ensure at least one A-Z
    # is present and no lowercase letters appear.
    return candidate.upper() == candidate and re.search(r"[A-Z]", candidate)


def _read_tokens(
    tsv_path: Path,
) -> Tuple[Dict[Tuple[int, int], List[Dict[str, str]]], List[Dict[str, Any]]]:
    """Read tokens from ``tsv_path`` grouped by `(page, line)`.

    Returns a tuple of ``(tokens_by_line, lines)`` where ``tokens_by_line`` maps
    `(page, line)` to the list of token dictionaries, and ``lines`` is an
    ordered list of consolidated line dictionaries containing ``page``,
    ``line`` and joined ``text``.
    """
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
                # Skip tokens with malformed page/line numbers
                continue
            tokens_by_line[(page, line)].append(row)

    lines: List[Dict[str, Any]] = []
    for page, line in sorted(tokens_by_line.keys()):
        text = "".join(tok.get("text", "") for tok in tokens_by_line[(page, line)])
        lines.append({"page": page, "line": line, "text": text})
    return tokens_by_line, lines


def _is_meaningful_heading(text: str) -> bool:
    """Return True if ``text`` looks like a useful heading.

    The line must be ALL-CAPS (per :func:`_is_heading`) and should not contain a
    colon, which often indicates a label rather than a heading.
    """
    return _is_heading(text) and ":" not in text


def _find_account_start(
    lines: List[Dict[str, Any]], anchor_idx: int
) -> Tuple[int, str | None, str]:
    """Return the start index, heading and heading_source for an account.

    ``anchor_idx`` points to the line containing ``Account #``. The function
    backtracks ``HEADING_BACK_LINES`` lines on the same page to locate the
    heading. If that fails, it tries progressively closer lines. As a smart
    fallback, if the chosen line is not a meaningful heading, search up to six
    lines back on the same page for one that is.
    """
    page = lines[anchor_idx]["page"]
    candidate = anchor_idx - HEADING_BACK_LINES
    if candidate < 0 or lines[candidate]["page"] != page:
        candidate = anchor_idx - 1
        if candidate < 0 or lines[candidate]["page"] != page:
            candidate = anchor_idx - 1 if anchor_idx > 0 else anchor_idx

    heading_guess = lines[candidate]["text"].strip() if candidate >= 0 else None
    heading_source = "pure_heading"
    if not (heading_guess and _is_meaningful_heading(heading_guess)):
        for back in range(1, 6 + 1):
            j = anchor_idx - back
            if j < 0 or lines[j]["page"] != page:
                break
            text = lines[j]["text"].strip()
            if _is_meaningful_heading(text):
                candidate = j
                heading_guess = text
                heading_source = "backtrack"
                break

    return max(candidate, 0), heading_guess, heading_source


def _find_heading_after_section(
    lines: List[Dict[str, Any]], section_idx: int
) -> Tuple[int, str | None]:
    """Return the index and heading after a section starter line."""

    for j in range(section_idx + 1, min(section_idx + 5, len(lines))):
        text = lines[j]["text"].strip()
        if not _is_meaningful_heading(text):
            continue
        if _norm(text) == BUREAUS_NORM:
            continue
        return j, text

    if section_idx + 1 < len(lines):
        return section_idx + 1, lines[section_idx + 1]["text"].strip()
    return section_idx + 1, None


def _write_account_tsv(
    out_dir: Path,
    account_index: int,
    account_lines: Iterable[Dict[str, Any]],
    tokens_by_line: Dict[Tuple[int, int], List[Dict[str, str]]],
) -> None:
    """Write a debug TSV for a single account."""
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"_debug_account_{account_index}.tsv"
    header = ["page", "line", "y0", "y1", "x0", "x1", "text"]
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for line in account_lines:
            key = (line["page"], line["line"])
            for tok in tokens_by_line.get(key, []):
                fh.write("\t".join(tok.get(h, "") for h in header) + "\n")


def split_accounts(
    tsv_path: Path, json_out: Path, write_tsv: bool = False
) -> Dict[str, Any]:
    """Core logic for splitting accounts from the full TSV."""
    tokens_by_line, lines = _read_tokens(tsv_path)

    stop_marker_seen = False
    for i, line in enumerate(lines):
        if _norm(line["text"]) == STOP_MARKER_NORM:
            stop_marker_seen = True
            lines = lines[:i]
            break

    anchors = [i for i, line in enumerate(lines) if ACCOUNT_RE.search(line["text"])]

    account_starts: List[int] = []
    headings: List[str | None] = []
    heading_sources: List[str] = []
    for anchor in anchors:
        start_idx, heading, source = _find_account_start(lines, anchor)
        account_starts.append(start_idx)
        headings.append(heading)
        heading_sources.append(source)

    section_starts = [i for i, line in enumerate(lines) if _norm(line["text"]) in SECTION_STARTERS]
    section_prefix_flags = [False] * len(account_starts)
    sections: List[str | None] = [None] * len(account_starts)

    for s_idx in section_starts:
        next_idx = bisect_right(account_starts, s_idx)
        if next_idx >= len(account_starts):
            continue
        section_prefix_flags[next_idx] = True
        heading_idx, heading = _find_heading_after_section(lines, s_idx)
        account_starts[next_idx] = heading_idx
        headings[next_idx] = heading
        heading_sources[next_idx] = "section+heading"
        starter_norm = _norm(lines[s_idx]["text"])
        sections[next_idx] = _SECTION_NAME.get(starter_norm)

    accounts: List[Dict[str, Any]] = []
    current_section: str | None = None
    section_ptr = 0
    for idx, start_idx in enumerate(account_starts):
        if sections[idx] is not None:
            current_section = sections[idx]
        sections[idx] = current_section
        next_start = (
            account_starts[idx + 1] if idx + 1 < len(account_starts) else len(lines)
        )
        while section_ptr < len(section_starts) and section_starts[section_ptr] < start_idx:
            section_ptr += 1
        cut_end = next_start
        if section_ptr < len(section_starts) and start_idx <= section_starts[section_ptr] < next_start:
            cut_end = section_starts[section_ptr]
        account_lines = lines[start_idx:cut_end]
        if not account_lines:
            continue
        account_info = {
            "account_index": idx + 1,
            "page_start": account_lines[0]["page"],
            "line_start": account_lines[0]["line"],
            "page_end": account_lines[-1]["page"],
            "line_end": account_lines[-1]["line"],
            "heading_guess": headings[idx],
            "heading_source": heading_sources[idx],
            "section": sections[idx],
            "section_prefix_seen": section_prefix_flags[idx],
            "lines": account_lines,
        }
        accounts.append(account_info)
        if write_tsv:
            _write_account_tsv(
                tsv_path.parent / "per_account_tsv",
                idx + 1,
                account_lines,
                tokens_by_line,
            )

    result = {"accounts": accounts, "stop_marker_seen": stop_marker_seen}
    json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Split accounts from the full TSV")
    ap.add_argument("--full", default="_debug_full.tsv", help="Input TSV path")
    ap.add_argument(
        "--json_out", default="accounts_from_full.json", help="JSON output path"
    )
    ap.add_argument(
        "--write-tsv",
        action="store_true",
        help="Write per-account TSVs to per_account_tsv/",
    )
    args = ap.parse_args(argv)

    tsv_path = Path(args.full)
    json_out = Path(args.json_out)
    split_accounts(tsv_path, json_out, write_tsv=args.write_tsv)
    print(f"Wrote accounts to {json_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
