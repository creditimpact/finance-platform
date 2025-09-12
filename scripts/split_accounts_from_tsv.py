#!/usr/bin/env python3
"""Split accounts from a full token TSV dump.

This script groups tokens by `(page, line)` to form lines of text. It detects
account boundaries based on lines that contain the exact string ``Account #``.
Each account is emitted to a structured JSON file and, optionally, into
individual TSV files for debugging.
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
from backend.core.logic.report_analysis.canonical_labels import LABEL_MAP
from backend.core.logic.report_analysis.normalize_fields import ensure_all_keys
from backend.core.logic.report_analysis.triad_layout import (
    EDGE_EPS,
    TriadLayout,
    assign_band,
    bands_from_header_tokens,
)

logger = logging.getLogger(__name__)
# Enable with RAW_TRIAD_FROM_X=1 for verbose triad logs
triad_log = logger.info if RAW_TRIAD_FROM_X else (lambda *a, **k: None)
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


_SPACE_RE = re.compile(r"\s+")


def join_tokens_with_space(tokens: Iterable[str]) -> str:
    """Join tokens with a single space, normalizing whitespace."""
    s = " ".join(t.strip() for t in tokens if t is not None)
    return _SPACE_RE.sub(" ", s).strip()


def _norm(text: str) -> str:
    """Normalize ``text`` by removing spaces/symbols and lowering case."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _norm_text(s: str) -> str:
    """Normalize text for guard checks by collapsing whitespace and punctuation."""
    s = s.replace("\u00ae", " ")
    s = s.replace(",", " ").replace(":", " ")
    return " ".join(s.split()).lower()


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


def is_account_anchor(joined_text: str) -> bool:
    """Return True if ``joined_text`` matches the exact ``Account #`` anchor."""
    # "Account" without the trailing ``#`` must not trigger activation
    return joined_text.strip().startswith("Account #")


def _looks_like_headline(text: str) -> bool:
    """Return True if ``text`` is an ALL-CAPS headline candidate."""
    stripped = re.sub(r"[^A-Za-z0-9/&\- ]", "", text).strip()
    if ":" in stripped:
        return False
    core = stripped.replace(" ", "")
    if len(core) < 3:
        return False
    return core.isupper()


def _mid_y(t: dict) -> float:
    """Return the vertical midpoint of token ``t``."""
    try:
        y0 = float(t.get("y0", 0.0))
        y1 = float(t.get("y1", y0))
        return (y0 + y1) / 2.0
    except Exception:
        return 0.0


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
    anchor_page: int,
    anchor_line: int,
    anchor_y: float,
) -> List[dict] | None:
    """Return tokens for the nearest header line strictly above the anchor.
    Header must contain exactly {'transunion','experian','equifax'} (any order),
    and nothing else after normalization.
    """

    def _is_pure_triad_header(joined: str) -> bool:
        s = _norm_text(joined)
        parts = s.split()
        names = {"transunion", "experian", "equifax"}
        return (
            len(parts) == 3 and set(parts) == names
        )  # exactly three tokens, no extras

    def _line_header(p: int, ln: int) -> Tuple[List[dict] | None, str | None]:
        toks = tokens_by_line.get((p, ln))
        if not toks:
            return None, None
        # ensure all tokens are above anchor_y
        if any(_mid_y(t) >= anchor_y for t in toks):
            return None, None
        joined = join_tokens_with_space([t.get("text", "") for t in toks])
        if not _is_pure_triad_header(joined):
            return None, None
        return toks, _norm_text(joined)

    # scan upwards on same page
    for ln in range(anchor_line - 1, 0, -1):
        header, norm = _line_header(anchor_page, ln)
        if header:
            ys = sorted(_mid_y(t) for t in header)
            yval = ys[len(ys) // 2] if ys else 0.0
            triad_log(
                "TRIAD_HEADER_ABOVE page=%s line=%s y=%.1f norm=%r",
                anchor_page,
                ln,
                yval,
                norm,
            )
            return header

    # check previous page from last line upwards
    prev_page = anchor_page - 1
    if prev_page >= 1:
        prev_lines = [ln for (pg, ln) in tokens_by_line.keys() if pg == prev_page]
        for ln in sorted(prev_lines, reverse=True):
            toks = tokens_by_line.get((prev_page, ln))
            if not toks:
                continue
            joined = join_tokens_with_space([t.get("text", "") for t in toks])
            if _is_pure_triad_header(joined):
                ys = sorted(_mid_y(t) for t in toks)
                yval = ys[len(ys) // 2] if ys else 0.0
                triad_log(
                    "TRIAD_HEADER_ABOVE page=%s line=%s y=%.1f norm=%r",
                    prev_page,
                    ln,
                    yval,
                    _norm_text(joined),
                )
                return toks

    triad_log(
        "TRIAD_NO_HEADER_ABOVE_ANCHOR page=%s line=%s",
        anchor_page,
        anchor_line,
    )
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

    stop_marker_seen = False
    for i, line in enumerate(lines):
        if _norm(line["text"]) == STOP_MARKER_NORM:
            stop_marker_seen = True
            lines = lines[:i]
            break

    anchors = [i for i, line in enumerate(lines) if is_account_anchor(line["text"])]

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

            def reset() -> None:
                nonlocal triad_active, current_layout, current_layout_page, open_row
                triad_active = False
                current_layout = None
                current_layout_page = None
                open_row = None

            for line_idx, line in enumerate(account_lines):
                key = (line["page"], line["line"])
                toks = tokens_by_line.get(key, [])
                texts = [t.get("text", "") for t in toks]
                joined_line_text = join_tokens_with_space(texts)

                s = _norm_text(joined_line_text)
                if (
                    triad_active
                    and current_layout
                    and current_layout_page is not None
                    and line["page"] != current_layout_page
                ):
                    prev_page = current_layout_page
                    triad_log(
                        "TRIAD_CARRY_PAGE from=%s to=%s",
                        prev_page,
                        line["page"],
                    )
                    current_layout_page = line["page"]

                is_heading_line_without_values = (
                    triad_active
                    and current_layout is not None
                    and not any(
                        assign_band(t, current_layout) in {"tu", "xp", "eq"}
                        for t in toks
                    )
                )
                if triad_active:
                    if s in {"two year payment history"}:
                        triad_log(
                            "TRIAD_STOP reason=two_year_history page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        reset()
                        continue
                    if s.startswith("days late - 7 year history"):
                        triad_log(
                            "TRIAD_STOP reason=days_late page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        reset()
                        continue
                    if (
                        s in {"transunion", "experian", "equifax"}
                        and is_heading_line_without_values
                    ):
                        triad_log(
                            "TRIAD_STOP reason=bare_bureau_header page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        reset()
                        continue
                    if is_account_anchor(joined_line_text):
                        triad_log(
                            "TRIAD_RESET_ON_ANCHOR page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        carry_over = account_lines[line_idx:]
                        account_lines = account_lines[:line_idx]
                        reset()
                        trailing_pruned = True
                        break

                layout: TriadLayout | None = None
                if not triad_active and is_account_anchor(joined_line_text):
                    toks_anchor = tokens_by_line.get((line["page"], line["line"]), [])
                    ys = sorted(_mid_y(t) for t in toks_anchor)
                    anchor_y = ys[len(ys) // 2] if ys else 0.0
                    triad_log(
                        "TRIAD_ANCHOR_AT page=%s line=%s y=%.1f",
                        line["page"],
                        line["line"],
                        anchor_y,
                    )
                    header_toks = find_header_above(
                        tokens_by_line,
                        line["page"],
                        line["line"],
                        anchor_y,
                    )
                    if header_toks:
                        layout = bands_from_header_tokens(header_toks)
                        triad_log(
                            "TRIAD_HEADER_XMIDS tu=%.1f xp=%.1f eq=%.1f",
                            layout.tu_band[0] + EDGE_EPS,
                            layout.xp_band[0] + EDGE_EPS,
                            layout.eq_band[0] + EDGE_EPS,
                        )
                        band_tokens_anchor = {"label": [], "tu": [], "xp": [], "eq": []}
                        for t in toks_anchor:
                            b = assign_band(t, layout)
                            if b in band_tokens_anchor:
                                band_tokens_anchor[b].append(t)
                        if band_tokens_anchor["label"] and all(
                            len(band_tokens_anchor[b]) == 1 for b in ("tu", "xp", "eq")
                        ):
                            triad_active = True
                            current_layout = layout
                            current_layout_page = line["page"]
                        else:
                            triad_log(
                                "TRIAD_STOP reason=layout_mismatch page=%s line=%s",
                                line["page"],
                                line["line"],
                            )
                            triad_active = False
                            current_layout = None
                            current_layout_page = None
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
                has_tu = bool(tu_val)
                has_xp = bool(xp_val)
                has_eq = bool(eq_val)

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
                if label_txt and (label_txt.endswith(":") or label_txt.endswith("#")):
                    visual = label_txt.rstrip(":").strip()
                    canonical_key = LABEL_MAP.get(visual)
                    if canonical_key is None and visual != "Account #":
                        triad_log(
                            "TRIAD_GUARD_SKIP page=%s line=%s reason=unknown_label label=%r",
                            line["page"],
                            line["line"],
                            visual,
                        )
                        open_row = None
                        continue
                    row = {
                        "triad_row": True,
                        "label": visual,
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
                        if not (has_tu or has_xp or has_eq):
                            triad_log(
                                "TRIAD_GUARD_SKIP page=%s line=%s reason=no_banded_tokens",
                                line["page"],
                                line["line"],
                            )
                            open_row = None
                            continue
                        if has_tu:
                            open_row["values"][
                                "transunion"
                            ] = f"{open_row['values']['transunion']} {tu_val}".strip()
                            if open_row["key"]:
                                triad_maps["transunion"][open_row["key"]] = open_row[
                                    "values"
                                ]["transunion"]
                        if has_xp:
                            open_row["values"][
                                "experian"
                            ] = f"{open_row['values']['experian']} {xp_val}".strip()
                            if open_row["key"]:
                                triad_maps["experian"][open_row["key"]] = open_row[
                                    "values"
                                ]["experian"]
                        if has_eq:
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
                            has_tu,
                            has_xp,
                            has_eq,
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
