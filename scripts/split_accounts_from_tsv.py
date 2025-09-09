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

ACCOUNT_RE = re.compile(r"\bAccount\b.*#", re.IGNORECASE)


def _is_heading(text: str) -> bool:
    """Return True if ``text`` appears to be an ALL-CAPS heading."""
    candidate = text.strip()
    if not candidate:
        return False
    # ``str.isupper`` handles alphanumeric with punctuation reasonably well but
    # fails on strings without alphabetic characters. Ensure at least one A-Z
    # is present and no lowercase letters appear.
    return candidate.upper() == candidate and re.search(r"[A-Z]", candidate)


def _read_tokens(tsv_path: Path) -> Tuple[Dict[Tuple[int, int], List[Dict[str, str]]], List[Dict[str, Any]]]:
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
    for (page, line) in sorted(tokens_by_line.keys()):
        text = "".join(tok.get("text", "") for tok in tokens_by_line[(page, line)])
        lines.append({"page": page, "line": line, "text": text})
    return tokens_by_line, lines


def _guess_heading(lines: List[Dict[str, Any]], idx: int) -> str | None:
    """Guess the heading for the account starting at ``lines[idx]``."""
    if idx == 0:
        return None
    prev_text = lines[idx - 1]["text"].strip()
    return prev_text if _is_heading(prev_text) else None


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
) -> List[Dict[str, Any]]:
    """Core logic for splitting accounts from the full TSV."""
    tokens_by_line, lines = _read_tokens(tsv_path)

    # Precompute which lines look like headings for later boundary adjustments
    heading_flags = [_is_heading(line["text"]) for line in lines]
    account_starts = [i for i, line in enumerate(lines) if ACCOUNT_RE.search(line["text"])]

    accounts: List[Dict[str, Any]] = []
    for idx, start_idx in enumerate(account_starts):
        next_start = account_starts[idx + 1] if idx + 1 < len(account_starts) else len(lines)
        end_idx = next_start - 1
        # If the line immediately before the next account looks like a heading,
        # exclude it from the current account.
        if idx + 1 < len(account_starts) and heading_flags[next_start - 1]:
            end_idx = next_start - 2
        account_lines = lines[start_idx : end_idx + 1]
        if not account_lines:
            continue
        account_info = {
            "account_index": idx + 1,
            "page_start": account_lines[0]["page"],
            "line_start": account_lines[0]["line"],
            "page_end": account_lines[-1]["page"],
            "line_end": account_lines[-1]["line"],
            "heading_guess": _guess_heading(lines, start_idx),
            "lines": account_lines,
        }
        accounts.append(account_info)
        if write_tsv:
            _write_account_tsv(tsv_path.parent / "per_account_tsv", idx + 1, account_lines, tokens_by_line)

    json_out.write_text(json.dumps(accounts, indent=2), encoding="utf-8")
    return accounts


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Split accounts from the full TSV")
    ap.add_argument("--full", default="_debug_full.tsv", help="Input TSV path")
    ap.add_argument("--json_out", default="accounts_from_full.json", help="JSON output path")
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
