from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from backend.core.logic.report_analysis.report_parsing import (
    build_block_fuzzy,
    detect_bureau_order,
)
from backend.core.logic.report_analysis.text_provider import load_cached_text
from backend.core.logic.utils.text_parsing import extract_account_blocks


def load_account_blocks(session_id: str) -> List[Dict[str, Any]]:
    """Load previously exported account blocks for ``session_id``.

    The blocks are expected under ``traces/blocks/<session_id>/_index.json``.
    The index must be a JSON array where each element is a mapping with
    exactly the keys ``{"i", "heading", "file"}``. If the directory or index
    file is missing, an entry is malformed, or a referenced block file cannot
    be read/parsed, the function fails softly and simply returns an empty
    list.

    Parameters
    ----------
    session_id:
        Identifier used for locating ``traces/blocks/<session_id>``.

    Returns
    -------
    list[dict]
        List of block dictionaries of the form ``{"heading": str,
        "lines": list[str]}``.
    """

    base = Path("traces") / "blocks" / session_id
    index_path = base / "_index.json"
    if not index_path.exists():
        return []
    try:
        idx = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    blocks: List[Dict[str, Any]] = []
    expected_keys = {"i", "heading", "file"}
    for entry in idx or []:
        if not isinstance(entry, dict):
            continue
        if set(entry.keys()) != expected_keys:
            # ignore unexpected/legacy index rows
            continue
        f = entry.get("file")
        if not isinstance(f, str) or not f:
            continue
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
            if isinstance(data, dict) and "heading" in data and "lines" in data:
                blocks.append(data)
        except Exception:
            continue
    return blocks


logger = logging.getLogger(__name__)


ENRICH_ENABLED = os.getenv("BLOCK_ENRICH", "1") != "0"


FIELD_LABELS: dict[str, str] = {
    "account #": "account_number_display",
    "high balance": "high_balance",
    "last verified": "last_verified",
    "date of last activity": "date_of_last_activity",
    "date reported": "date_reported",
    "date opened": "date_opened",
    "balance owed": "balance_owed",
    "closed date": "closed_date",
    "account rating": "account_rating",
    "account description": "account_description",
    "dispute status": "dispute_status",
    "creditor type": "creditor_type",
    "account status": "account_status",
    "payment status": "payment_status",
    "creditor remarks": "creditor_remarks",
    "payment amount": "payment_amount",
    "last payment": "last_payment",
    "term length": "term_length",
    "past due amount": "past_due_amount",
    "account type": "account_type",
    "payment frequency": "payment_frequency",
    "credit limit": "credit_limit",
}


def _split_vals(text: str, parts: int) -> list[str]:
    """Split ``text`` into ``parts`` values using column heuristics."""

    if not text:
        return [""] * parts

    vals = re.split(r"\s{2,}", text.strip())
    if len(vals) != parts:
        tokens = text.strip().split()
        if len(tokens) >= parts:
            vals = tokens[: parts - 1] + [" ".join(tokens[parts - 1 :])]
        else:
            vals = tokens + [""] * (parts - len(tokens))
    if len(vals) > parts:
        vals = vals[: parts - 1] + [" ".join(vals[parts - 1 :])]
    if len(vals) < parts:
        vals += [""] * (parts - len(vals))
    return [v.strip() for v in vals]


def enrich_block(blk: dict) -> dict:
    """Add structured ``fields`` map parsed from ``blk['lines']``."""

    heading = blk.get("heading", "")
    logger.warning("ENRICH: start heading=%r", heading)

    order = detect_bureau_order(blk.get("lines") or [])
    if not order:
        raise ValueError("Block has no bureau columns")

    # initialise fields map with empty strings
    field_keys = list(FIELD_LABELS.values())
    fields = {
        b: {k: "" for k in field_keys} for b in ["transunion", "experian", "equifax"]
    }

    in_section = False
    for line in blk.get("lines") or []:
        clean = line.strip()
        if not clean:
            continue
        if not in_section:
            norm = re.sub(r"[^a-z]+", " ", clean.lower())
            if all(b in norm for b in order):
                in_section = True
            continue

        norm_line = clean.lower()
        for label, key in FIELD_LABELS.items():
            if norm_line.startswith(label):
                rest = clean[len(label) :].strip()
                if rest.startswith(":"):
                    rest = rest[1:].strip()
                vals = _split_vals(rest, len(order))
                for idx, bureau in enumerate(order):
                    v = vals[idx] if idx < len(vals) else ""
                    v = v if v not in {"--", "-"} else ""
                    fields[bureau][key] = v
                break

    tu_count = sum(1 for v in fields["transunion"].values() if v)
    ex_count = sum(1 for v in fields["experian"].values() if v)
    eq_count = sum(1 for v in fields["equifax"].values() if v)
    logger.warning(
        "ENRICH: fields_done tu=%d ex=%d eq=%d", tu_count, ex_count, eq_count
    )
    logger.info(
        "BLOCK: enrichment_summary heading=%r tu_filled=%d ex_filled=%d eq_filled=%d",
        heading,
        tu_count,
        ex_count,
        eq_count,
    )

    return {**blk, "fields": fields}


def export_account_blocks(
    session_id: str, pdf_path: str | Path
) -> List[Dict[str, Any]]:
    """Extract account blocks from ``pdf_path`` and export them to JSON files.

    Parameters
    ----------
    session_id:
        Identifier used for the output directory ``traces/blocks/<session_id>``.
    pdf_path:
        Path to the PDF to parse.

    Returns
    -------
    list[dict]
        The list of account block dictionaries, each containing ``heading`` and
        ``lines`` keys.
    """
    cached = load_cached_text(session_id)
    if not cached:
        raise ValueError("no_cached_text_for_session")
    text = cached["full_text"]
    blocks = extract_account_blocks(text)

    fbk_blocks: List[Dict[str, Any]] = []
    for blk in blocks:
        if not blk:
            continue
        heading = (blk[0] or "").strip()
        fbk_blocks.append({"heading": heading, "lines": blk})

    if not fbk_blocks:
        logger.error(
            "BLOCKS_FAIL_FAST: 0 blocks extracted sid=%s file=%s",
            session_id,
            str(pdf_path),
        )
        raise ValueError("No blocks extracted")

    blocks_by_account_fuzzy = build_block_fuzzy(fbk_blocks) if fbk_blocks else {}
    logger.warning(
        "ANZ: pre-save fbk=%d fuzzy=%d sid=%s",
        len(fbk_blocks),
        len(blocks_by_account_fuzzy or {}),
        session_id,
    )

    out_dir = Path("traces") / "blocks" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.warning("BLOCK_ENRICH: enabled=%s sid=%s", ENRICH_ENABLED, session_id)

    out_blocks: List[Dict[str, Any]] = []
    idx_info = []
    for i, blk in enumerate(fbk_blocks, 1):
        out_blk = enrich_block(blk) if ENRICH_ENABLED else blk
        out_blocks.append(out_blk)
        jpath = out_dir / f"block_{i:02d}.json"
        with jpath.open("w", encoding="utf-8") as f:
            json.dump(out_blk, f, ensure_ascii=False, indent=2)
        idx_info.append({"i": i, "heading": out_blk["heading"], "file": str(jpath)})

    with (out_dir / "_index.json").open("w", encoding="utf-8") as f:
        json.dump(idx_info, f, ensure_ascii=False, indent=2)

    logger.warning(
        "ANZ: export blocks sid=%s dir=%s files=%d",
        session_id,
        str(out_dir),
        len(out_blocks),
    )

    return out_blocks
