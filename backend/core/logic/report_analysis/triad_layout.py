from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from backend.config import RAW_TRIAD_FROM_X

from .header_utils import normalize_bureau_header

logger = logging.getLogger(__name__)
triad_log = logger.info if RAW_TRIAD_FROM_X else (lambda *a, **k: None)

# Tolerance applied to both sides of each band when assigning tokens. This
# guards against OCR jitter around column boundaries so tokens landing slightly
# outside the band are still classified correctly.
EDGE_EPS = 6.0


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


def assign_band(
    token: dict, layout: TriadLayout
) -> Literal["label", "tu", "xp", "eq", "none"]:
    """Assign a token to one of the triad bands.

    Tokens are classified by comparing their midpoint against each band's left
    and right edges with a small symmetric tolerance. This avoids spurious
    ``none`` classifications for tokens that land on the seam between columns.
    """
    x = mid_x(token)
    bands = {
        "label": layout.label_band,
        "tu": layout.tu_band,
        "xp": layout.xp_band,
        "eq": layout.eq_band,
    }
    for name, (L, R) in bands.items():
        if L - EDGE_EPS <= x <= R + EDGE_EPS:
            return name
    return "none"


def detect_triads(
    tokens_by_line: Dict[Tuple[int, int], List[dict]]
) -> Dict[int, TriadLayout]:
    """Detect per-page triad layouts from token lines."""
    by_page: Dict[int, Dict[int, List[dict]]] = {}
    for (page, line), toks in tokens_by_line.items():
        by_page.setdefault(page, {})[line] = toks

    layouts: Dict[int, TriadLayout] = {}
    for page, lines in by_page.items():
        mids: Dict[str, float] | None = None
        for line_no, toks in sorted(lines.items()):
            found: Dict[str, dict] = {}
            for t in toks:
                tnorm = normalize_bureau_header(str(t.get("text", "")))
                if (
                    tnorm in {"transunion", "experian", "equifax"}
                    and tnorm not in found
                ):
                    found[tnorm] = t
            if len(found) == 3:
                mids = {k: mid_x(v) for k, v in found.items()}
                break
        if not mids or len(mids) != 3:
            continue
        tu = mids["transunion"]
        xp = mids["experian"]
        eq = mids["equifax"]
        d12 = xp - tu
        d23 = eq - xp
        label_band = (0.0, tu - d12 / 2.0)
        tu_band = (tu - d12 / 2.0, tu + d12 / 2.0)
        xp_band = (tu + d12 / 2.0, xp + d23 / 2.0)
        eq_band = (xp + d23 / 2.0, eq + d23 / 2.0)
        layout = TriadLayout(
            page=page,
            label_band=label_band,
            tu_band=tu_band,
            xp_band=xp_band,
            eq_band=eq_band,
        )
        layouts[page] = layout
        triad_log(
            "TRIAD_LAYOUT page=%s label=(%.1f,%.1f) tu=(%.1f,%.1f) xp=(%.1f,%.1f) eq=(%.1f,%.1f)",
            page,
            label_band[0],
            label_band[1],
            tu_band[0],
            tu_band[1],
            xp_band[0],
            xp_band[1],
            eq_band[0],
            eq_band[1],
        )
    return layouts
