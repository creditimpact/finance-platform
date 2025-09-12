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


def bands_from_header_tokens(tokens: List[dict]) -> TriadLayout:
    """Compute a ``TriadLayout`` from three bureau header tokens.

    ``tokens`` must contain the ``transunion``, ``experian`` and ``equifax``
    headers from the same line (in any order). Their horizontal midpoints are
    computed and sorted left-to-right to establish band boundaries. The
    ``label`` band spans from ``0`` to the leftmost midpoint. Each bureau band
    then covers ``[mid_i, mid_{i+1})`` where the final band extends to
    ``+inf``. ``assign_band`` applies the symmetric :data:`EDGE_EPS` tolerance
    when classifying tokens against these bands.
    """

    mids: List[Tuple[float, str]] = []
    page = 0
    for t in tokens:
        name = normalize_bureau_header(str(t.get("text", "")))
        if name not in {"transunion", "experian", "equifax"}:
            continue
        mids.append((mid_x(t), name))
        if not page:
            try:
                page = int(float(t.get("page", 0)))
            except Exception:
                page = 0
    if len(mids) != 3:
        raise ValueError("expected three bureau header tokens")

    mids.sort(key=lambda kv: kv[0])
    label_band = (0.0, mids[0][0])

    bands: Dict[str, Tuple[float, float]] = {}
    for idx, (mid, name) in enumerate(mids):
        right = mids[idx + 1][0] if idx + 1 < len(mids) else float("inf")
        bands[name] = (mid, right)

    return TriadLayout(
        page=page,
        label_band=label_band,
        tu_band=bands["transunion"],
        xp_band=bands["experian"],
        eq_band=bands["equifax"],
    )


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
        layout: TriadLayout | None = None
        for _line_no, toks in sorted(lines.items()):
            found: Dict[str, dict] = {}
            for t in toks:
                raw_text = str(t.get("text", ""))
                tnorm = normalize_bureau_header(raw_text)
                if tnorm in {"transunion", "experian", "equifax"}:
                    triad_log("TRIAD_HEADER_MATCH raw=%r norm=%r", raw_text, tnorm)
                    if tnorm not in found:
                        found[tnorm] = t
            if len(found) == 3:
                layout = bands_from_header_tokens(list(found.values()))
                layout.page = page
                break
        if not layout:
            continue
        layouts[page] = layout
        triad_log(
            "TRIAD_LAYOUT page=%s label=(%.1f,%.1f) tu=(%.1f,%.1f) xp=(%.1f,%.1f) eq=(%.1f,%.1f)",
            layout.page,
            layout.label_band[0],
            layout.label_band[1],
            layout.tu_band[0],
            layout.tu_band[1],
            layout.xp_band[0],
            layout.xp_band[1],
            layout.eq_band[0],
            layout.eq_band[1],
        )
    return layouts
