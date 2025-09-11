from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class TriadLayout:
    page: int
    label_band: Tuple[float, float]
    tu_band: Tuple[float, float]
    xp_band: Tuple[float, float]
    eq_band: Tuple[float, float]


def mid_x(tok: dict) -> float:
    try:
        x0 = float(tok.get("x0", 0.0))
        x1 = float(tok.get("x1", x0))
        return (x0 + x1) / 2.0
    except Exception:
        return 0.0


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def detect_triads(tokens_by_line: Dict[Tuple[int, int], List[dict]]) -> Dict[int, TriadLayout]:
    """Detect per-page triad layouts from token lines."""
    by_page: Dict[int, Dict[int, List[dict]]] = {}
    for (page, line), toks in tokens_by_line.items():
        by_page.setdefault(page, {})[line] = toks

    layouts: Dict[int, TriadLayout] = {}
    for page, lines in by_page.items():
        header: List[dict] | None = None
        for line_no, toks in sorted(lines.items()):
            text = " ".join(t.get("text", "") for t in toks)
            if norm(text) == "transunion experian equifax":
                header = toks
                break
        if not header:
            continue

        mids: Dict[str, float] = {}
        for t in header:
            tnorm = norm(str(t.get("text", "")))
            if tnorm == "transunion":
                mids["tu"] = mid_x(t)
            elif tnorm == "experian":
                mids["xp"] = mid_x(t)
            elif tnorm == "equifax":
                mids["eq"] = mid_x(t)
        if len(mids) != 3:
            continue
        tu = mids["tu"]
        xp = mids["xp"]
        eq = mids["eq"]
        d12 = xp - tu
        d23 = eq - xp
        label_band = (0.0, tu - d12 / 2.0)
        tu_band = (tu - d12 / 2.0, tu + d12 / 2.0)
        xp_band = (tu + d12 / 2.0, xp + d23 / 2.0)
        eq_band = (xp + d23 / 2.0, eq + d23 / 2.0)
        layout = TriadLayout(page=page, label_band=label_band, tu_band=tu_band, xp_band=xp_band, eq_band=eq_band)
        layouts[page] = layout
        logger.info(
            "TRIAD_LAYOUT page=%s label=%s tu=%s xp=%s eq=%s",
            page,
            layout.label_band,
            layout.tu_band,
            layout.xp_band,
            layout.eq_band,
        )
    return layouts
