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
import logging
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from backend.config import RAW_JOIN_TOKENS_WITH_SPACE, RAW_TRIAD_FROM_X
from backend.core.logic.report_analysis.block_exporter import join_tokens_with_space
from backend.core.logic.report_analysis.canonical_labels import LABEL_MAP
from backend.core.logic.report_analysis.normalize_fields import ensure_all_keys
from backend.core.logic.report_analysis.report_parsing import ACCOUNT_NUMBER_ALIASES
from backend.core.logic.report_analysis.triad_layout import (
    TriadLayout,
    assign_band,
    bands_from_header_tokens,
    detect_triads,
)

logger = logging.getLogger(__name__)
# Enable with RAW_TRIAD_FROM_X=1 for verbose triad logs
triad_log = logger.info if RAW_TRIAD_FROM_X else (lambda *a, **k: None)

ACCOUNT_RE = re.compile(r"\bAccount\b.*#", re.IGNORECASE)
STOP_MARKER_NORM = "publicinformation"
SECTION_STARTERS = {"collection", "unknown"}
_SECTION_NAME = {"collection": "collections", "unknown": "unknown"}
NOISE_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
NOISE_BANNER_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4}.*(?:Credit\s*Report|SmartCredit)", re.IGNORECASE
)

# How many lines above the ``Account #`` anchor to consider the heading.
# Per hardening rules we look farther back to find a suitable headline.
HEADING_BACK_LINES = 8


def _norm(text: str) -> str:
    """Normalize ``text`` by removing spaces/symbols and lowering case."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


ACCOUNT_NUMBER_ALIAS_NORMS = {_norm(a) for a in ACCOUNT_NUMBER_ALIASES}


def _is_triad(text: str) -> bool:
    """Return True if ``text`` is the TransUnion/Experian/Equifax triad.

    The match is flexible and ignores punctuation such as ®, commas or
    colons. The three bureau names may appear in any order. When a match is
    detected a debug log is emitted for traceability.
    """
    s = (text or "").lower()
    s = s.replace("\u00ae", "")
    s = s.replace(",", " ").replace(":", " ")
    s = re.sub(r"\s+", " ", s).strip()
    hit = all(name in s for name in ("transunion", "experian", "equifax"))
    if hit:
        triad_log("TRIAD_HEADER_MATCH raw=%r norm=%r", text, s)
    return hit


def _is_anchor(text: str) -> bool:
    """Return True if ``text`` contains the literal ``Account #`` anchor."""
    return "account#" in _norm(text)


def is_account_anchor(text: str) -> bool:
    """Return True if ``text`` matches the ``Account #`` anchor exactly."""
    return text.strip() == "Account #"


def _looks_like_headline(text: str) -> bool:
    """Return True if ``text`` is an ALL-CAPS headline candidate."""
    stripped = re.sub(r"[^A-Za-z0-9/&\- ]", "", text).strip()
    if ":" in stripped:
        return False
    core = stripped.replace(" ", "")
    if len(core) < 3:
        return False
    return core.isupper()


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
        tokens = [tok.get("text", "") for tok in tokens_by_line[(page, line)]]
        if RAW_JOIN_TOKENS_WITH_SPACE:
            text = join_tokens_with_space(tokens)
        else:
            text = "".join(tokens)
        lines.append({"page": page, "line": line, "text": text})
    return tokens_by_line, lines


def find_header_above(
    tokens_by_line: Dict[Tuple[int, int], List[Dict[str, str]]],
    page: int,
    line: int,
) -> List[dict] | None:
    """Return header tokens (transunion/experian/equifax) on nearest line above.

    The search first checks ``(page, line-1)``. If that line is missing or does
    not contain the three bureau names, the last line of ``page-1`` is checked.
    Token text is normalized by stripping the registered mark, commas and
    colons, collapsing spaces and lowercasing. When a header is found a log
    entry ``TRIAD_HEADER_ABOVE`` is emitted, otherwise ``TRIAD_NO_HEADER_ABOVE_ANCHOR``
    is logged.
    """

    def _norm_header(text: str) -> str:
        s = (text or "").lower()
        s = s.replace("\u00ae", "").replace(",", " ").replace(":", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _extract_if_header(
        p: int, line_no: int
    ) -> Tuple[List[dict] | None, str | None]:
        toks = tokens_by_line.get((p, line_no))
        if not toks:
            return None, None
        joined = join_tokens_with_space([t.get("text", "") for t in toks])
        norm = _norm_header(joined)
        if all(name in norm for name in ("transunion", "experian", "equifax")):
            result: List[dict] = []
            for t in toks:
                tnorm = _norm_header(t.get("text", ""))
                if tnorm in {"transunion", "experian", "equifax"}:
                    result.append(t)
            if len(result) == 3:
                return result, norm
        return None, None

    header, norm = _extract_if_header(page, line - 1)
    if not header and page > 1:
        prev_lines = [ln for (pg, ln) in tokens_by_line.keys() if pg == page - 1]
        if prev_lines:
            last_line = max(prev_lines)
            header, norm = _extract_if_header(page - 1, last_line)

    if header:
        triad_log("TRIAD_HEADER_ABOVE page=%s line=%s norm=%r", page, line, norm)
        return header

    triad_log("TRIAD_NO_HEADER_ABOVE_ANCHOR page=%s line=%s", page, line)
    return None


def _pick_headline(
    lines: List[Dict[str, Any]], anchor_idx: int, back: int = HEADING_BACK_LINES
) -> Tuple[int, str | None, str]:
    """Return `(start_idx, heading_guess, heading_source)` for an anchor."""

    page = lines[anchor_idx]["page"]

    def _iter_back(start: int):
        for j in range(start, max(anchor_idx - back, -1), -1):
            if lines[j]["page"] != page:
                break
            yield j

    triad_idx: int | None = None
    for j in _iter_back(anchor_idx - 1):
        txt = lines[j]["text"]
        if _is_triad(txt):
            triad_idx = j
            if (
                j - 1 >= 0
                and lines[j - 1]["page"] == page
                and not _is_anchor(lines[j - 1]["text"])
                and _looks_like_headline(lines[j - 1]["text"])
            ):
                return j - 1, lines[j - 1]["text"].strip(), "triad_above"
            break

    for j in _iter_back(anchor_idx - 1):
        txt = lines[j]["text"]
        if _is_anchor(txt) or _is_triad(txt):
            continue
        if _looks_like_headline(txt):
            return j, txt.strip(), "backtrack"

    start_idx = triad_idx if triad_idx is not None else anchor_idx
    return start_idx, None, "anchor_no_heading"


def _find_heading_after_section(
    lines: List[Dict[str, Any]], section_idx: int
) -> Tuple[int, str | None]:
    """Return the index and heading after a section starter line."""

    for j in range(section_idx + 1, min(section_idx + 5, len(lines))):
        text = lines[j]["text"].strip()
        if _is_triad(text) or not _looks_like_headline(text):
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
    layouts = detect_triads(tokens_by_line) if RAW_TRIAD_FROM_X else {}

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
        start_idx, heading, source = _pick_headline(lines, anchor)
        account_starts.append(start_idx)
        headings.append(heading)
        heading_sources.append(source)

    section_starts = [
        i for i, line in enumerate(lines) if _norm(line["text"]) in SECTION_STARTERS
    ]
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
    carry_over: List[Dict[str, Any]] = []
    for idx, start_idx in enumerate(account_starts):
        if sections[idx] is not None:
            current_section = sections[idx]
        sections[idx] = current_section
        next_start = (
            account_starts[idx + 1] if idx + 1 < len(account_starts) else len(lines)
        )
        while (
            section_ptr < len(section_starts)
            and section_starts[section_ptr] < start_idx
        ):
            section_ptr += 1
        cut_end = next_start
        trailing_pruned = False
        if (
            section_ptr < len(section_starts)
            and start_idx <= section_starts[section_ptr] < next_start
        ):
            cut_end = section_starts[section_ptr]
            trailing_pruned = True
        account_lines = carry_over + lines[start_idx:cut_end]
        carry_over = []
        noise_lines_skipped = 0
        filtered_lines: List[Dict[str, Any]] = []
        for line in account_lines:
            text = line["text"].strip()
            if NOISE_URL_RE.match(text) or NOISE_BANNER_RE.match(text):
                noise_lines_skipped += 1
                continue
            filtered_lines.append(line)
        account_lines = filtered_lines

        def _is_structural_marker(txt: str) -> bool:
            n = _norm(txt)
            return n in SECTION_STARTERS or _is_triad(txt) or _is_anchor(txt)

        while account_lines and _is_structural_marker(account_lines[-1]["text"]):
            carry_over.insert(0, account_lines.pop())
            trailing_pruned = True
        if not account_lines:
            continue
        triad_rows: List[Dict[str, Any]] = []
        triad_maps: Dict[str, Dict[str, str]] = {
            "transunion": {},
            "experian": {},
            "equifax": {},
        }
        if RAW_TRIAD_FROM_X:
            open_row: Dict[str, Any] | None = None
            triad_active: bool = False
            current_layout: TriadLayout | None = None
            current_layout_page: int | None = None
            for line_idx, line in enumerate(account_lines):
                key = (line["page"], line["line"])
                toks = tokens_by_line.get(key, [])
                texts = [t.get("text", "") for t in toks]
                joined_line_text = join_tokens_with_space(texts)

                s = _norm(joined_line_text)
                if (
                    triad_active
                    and current_layout
                    and current_layout_page is not None
                    and line["page"] != current_layout_page
                ):
                    triad_log(
                        "TRIAD_CARRY_PAGE from=%s to=%s",
                        current_layout_page,
                        line["page"],
                    )
                    current_layout_page = line["page"]

                if triad_active:
                    if s == "twoyearpaymenthistory":
                        triad_log(
                            "TRIAD_STOP reason=two_year_history page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        triad_active = False
                        current_layout = None
                        current_layout_page = None
                        open_row = None
                        continue
                    if s in {"transunion", "experian", "equifax"}:
                        triad_log(
                            "TRIAD_STOP reason=bare_bureau_header page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        triad_active = False
                        current_layout = None
                        current_layout_page = None
                        open_row = None
                        continue
                    if is_account_anchor(joined_line_text):
                        triad_log(
                            "TRIAD_RESET_ON_ANCHOR page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        carry_over = account_lines[line_idx:]
                        account_lines = account_lines[:line_idx]
                        triad_active = False
                        current_layout = None
                        current_layout_page = None
                        open_row = None
                        trailing_pruned = True
                        break
                    if s.startswith("dayslate7yearhistory"):
                        triad_log(
                            "TRIAD_STOP reason=dayslate7yearhistory page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        triad_active = False
                        current_layout = None
                        current_layout_page = None
                        open_row = None
                        continue

                layout: TriadLayout | None = None
                if not triad_active and is_account_anchor(joined_line_text):
                    triad_log(
                        "TRIAD_ANCHOR_AT page=%s line=%s",
                        line["page"],
                        line["line"],
                    )
                    header_toks = find_header_above(
                        tokens_by_line, line["page"], line["line"]
                    )
                    if header_toks:
                        layout = bands_from_header_tokens(header_toks)
                        triad_log(
                            "TRIAD_HEADER_XMIDS tu=%.1f xp=%.1f eq=%.1f",
                            layout.tu_band[0],
                            layout.xp_band[0],
                            layout.eq_band[0],
                        )
                        triad_active = True
                        current_layout = layout
                        current_layout_page = line["page"]
                elif _is_triad(joined_line_text):
                    layout = layouts.get(line["page"])
                    if layout:
                        triad_active = True
                        current_layout = layout
                        if current_layout_page != line["page"]:
                            triad_log("TRIAD_CARRY start page=%s", line["page"])
                        current_layout_page = line["page"]
                elif triad_active and current_layout:
                    layout = current_layout
                band_tokens: Dict[str, List[dict]] = {
                    "label": [],
                    "tu": [],
                    "xp": [],
                    "eq": [],
                }
                if layout:
                    for t in toks:
                        band = assign_band(t, layout)
                        if band in band_tokens:
                            band_tokens[band].append(t)
                    if (
                        triad_active
                        and ":" in joined_line_text
                        and not band_tokens["label"]
                    ):
                        triad_log(
                            "TRIAD_STOP reason=layout_mismatch page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        triad_active = False
                        current_layout = None
                        current_layout_page = None
                        open_row = None
                        continue
                label_txt = join_tokens_with_space(
                    [t.get("text", "") for t in band_tokens["label"]]
                ).strip()
                tu_val = join_tokens_with_space(
                    [t.get("text", "") for t in band_tokens["tu"]]
                ).strip()
                xp_val = join_tokens_with_space(
                    [t.get("text", "") for t in band_tokens["xp"]]
                ).strip()
                eq_val = join_tokens_with_space(
                    [t.get("text", "") for t in band_tokens["eq"]]
                ).strip()

                if not layout:
                    if open_row:
                        triad_log(
                            "TRIAD_GUARD_SKIP page=%s line=%s reason=%s",
                            line["page"],
                            line["line"],
                            "no_layout",
                        )
                        open_row = None
                    continue
                plain_label = _norm(label_txt)
                is_account_num_alias = plain_label in ACCOUNT_NUMBER_ALIAS_NORMS
                if label_txt and (
                    label_txt.endswith(":")
                    or label_txt.endswith("#")
                    or is_account_num_alias
                ):
                    label = label_txt.rstrip(":")
                    canonical_key = LABEL_MAP.get(label)
                    if is_account_num_alias or canonical_key == "account_number_display":
                        canonical_key = "account_number_display"
                    row = {
                        "triad_row": True,
                        "label": label,
                        "key": canonical_key,
                        "values": {
                            "transunion": "",
                            "experian": "",
                            "equifax": "",
                        },
                    }
                    triad_rows.append(row)
                    if tu_val:
                        row["values"]["transunion"] = tu_val
                        if canonical_key:
                            triad_maps["transunion"][canonical_key] = tu_val
                    if xp_val:
                        row["values"]["experian"] = xp_val
                        if canonical_key:
                            triad_maps["experian"][canonical_key] = xp_val
                    if eq_val:
                        row["values"]["equifax"] = eq_val
                        if canonical_key:
                            triad_maps["equifax"][canonical_key] = eq_val
                    triad_log(
                        "TRIAD_ROW key=%s TU=%r XP=%r EQ=%r",
                        canonical_key,
                        tu_val,
                        xp_val,
                        eq_val,
                    )
                    open_row = row
                else:
                    if open_row:
                        if not (triad_active and current_layout):
                            triad_log(
                                "TRIAD_GUARD_SKIP page=%s line=%s reason=%s",
                                line["page"],
                                line["line"],
                                "triad_inactive",
                            )
                            open_row = None
                            continue
                        if not (tu_val or xp_val or eq_val):
                            triad_log(
                                "TRIAD_GUARD_SKIP page=%s line=%s reason=no_banded_tokens",
                                line["page"],
                                line["line"],
                            )
                            open_row = None
                            continue
                        if tu_val:
                            open_row["values"][
                                "transunion"
                            ] = f"{open_row['values']['transunion']} {tu_val}".strip()
                            if open_row["key"]:
                                triad_maps["transunion"][open_row["key"]] = open_row[
                                    "values"
                                ]["transunion"]
                        if xp_val:
                            open_row["values"][
                                "experian"
                            ] = f"{open_row['values']['experian']} {xp_val}".strip()
                            if open_row["key"]:
                                triad_maps["experian"][open_row["key"]] = open_row[
                                    "values"
                                ]["experian"]
                        if eq_val:
                            open_row["values"][
                                "equifax"
                            ] = f"{open_row['values']['equifax']} {eq_val}".strip()
                            if open_row["key"]:
                                triad_maps["equifax"][open_row["key"]] = open_row[
                                    "values"
                                ]["equifax"]
                        triad_log(
                            "TRIAD_CONT_PARTIAL page=%s line=%s tu=%s xp=%s eq=%s",
                            line["page"],
                            line["line"],
                            bool(tu_val),
                            bool(xp_val),
                            bool(eq_val),
                        )
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
            "trailing_section_marker_pruned": trailing_pruned,
            "noise_lines_skipped": noise_lines_skipped,
        }
        if RAW_TRIAD_FROM_X:
            account_info["triad"] = {
                "enabled": True,
                "order": ["transunion", "experian", "equifax"],
            }
            account_info["triad_fields"] = {
                "transunion": ensure_all_keys(triad_maps["transunion"]),
                "experian": ensure_all_keys(triad_maps["experian"]),
                "equifax": ensure_all_keys(triad_maps["equifax"]),
            }
            account_info["triad_rows"] = triad_rows
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
    ap.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a summary of accounts per section",
    )
    args = ap.parse_args(argv)

    tsv_path = Path(args.full)
    json_out = Path(args.json_out)
    result = split_accounts(tsv_path, json_out, write_tsv=args.write_tsv)
    if args.print_summary:
        accounts = result.get("accounts") or []
        total = len(accounts)
        collections = sum(1 for a in accounts if a.get("section") == "collections")
        unknown = sum(1 for a in accounts if a.get("section") == "unknown")
        regular = total - collections - unknown
        bad_last = [
            a["account_index"]
            for a in accounts
            if a.get("lines") and _norm(a["lines"][-1]["text"]) in SECTION_STARTERS
        ]
        print(f"Total accounts: {total}")
        print(f"collections: {collections} unknown: {unknown} regular: {regular}")
        if bad_last:
            print(f"Accounts ending with section starter: {bad_last}")
    print(f"Wrote accounts to {json_out}")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main()
