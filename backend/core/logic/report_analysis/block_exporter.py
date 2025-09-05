from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from backend.core.logic.report_analysis.report_parsing import (
    build_block_fuzzy,
    extract_text_from_pdf,
)
from backend.core.logic.utils.text_parsing import extract_account_blocks


def load_account_blocks(session_id: str) -> List[Dict[str, Any]]:
    """Load previously exported account blocks for ``session_id``.

    The blocks are expected under ``traces/blocks/<session_id>/_index.json``.
    If the directory or index file is missing, or any individual block file
    cannot be read/parsed, the function fails softly and simply returns an
    empty list.

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
    for entry in idx or []:
        f = entry.get("file")
        if not f:
            continue
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
            if isinstance(data, dict) and "heading" in data and "lines" in data:
                blocks.append(data)
        except Exception:
            continue
    return blocks

logger = logging.getLogger(__name__)


def export_account_blocks(session_id: str, pdf_path: str | Path) -> List[Dict[str, Any]]:
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
    text = extract_text_from_pdf(pdf_path)
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
        "ANZ: pre-save fbk=%d fuzzy=%d sid=%s req=%s",
        len(fbk_blocks),
        len((blocks_by_account_fuzzy or {}).keys()),
        session_id,
        None,
    )

    out_dir = Path("traces") / "blocks" / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_info = []
    for i, blk in enumerate(fbk_blocks, 1):
        jpath = out_dir / f"block_{i:02d}.json"
        with jpath.open("w", encoding="utf-8") as f:
            json.dump(blk, f, ensure_ascii=False, indent=2)
        idx_info.append({"i": i, "heading": blk["heading"], "file": str(jpath)})

    with (out_dir / "_index.json").open("w", encoding="utf-8") as f:
        json.dump(idx_info, f, ensure_ascii=False, indent=2)

    logger.warning(
        "ANZ: export blocks sid=%s dir=%s files=%d",
        session_id,
        str(out_dir),
        len(fbk_blocks),
    )

    return fbk_blocks
