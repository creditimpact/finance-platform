from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from backend.config import RAW_TRIAD_FROM_X

from .header_utils import normalize_bureau_header

logger = logging.getLogger(__name__)
triad_log = logger.info if RAW_TRIAD_FROM_X else (lambda *a, **k: None)

# Tolerance (in PDF points) applied around the TransUnion midpoint when
# computing the label/TU boundary. This guards against OCR jitter around the
# left edge so tokens landing slightly outside the column are still classified
# correctly.
EDGE_EPS = 6.0
EDGE_EPS_LABEL = 9.0


@dataclass
class TriadLayout:
    page: int
    label_band: Tuple[float, float]
    tu_band: Tuple[float, float]
    xp_band: Tuple[float, float]
    eq_band: Tuple[float, float]
    # Optional x0-based cutoffs (used when TRIAD_BAND_BY_X0=1)
    label_right_x0: float = 0.0
    tu_left_x0: float = 0.0
    xp_left_x0: float = 0.0
    eq_left_x0: float = 0.0


def bands_from_header_tokens(tokens: List[dict]) -> TriadLayout:
    """Compute a :class:`TriadLayout` from three bureau header tokens.

    ``tokens`` must contain exactly the ``transunion``, ``experian`` and
    ``equifax`` headers from the same line (in any order). Their horizontal
    midpoints are computed and non-overlapping bands are derived by splitting at
    midpoints between bureaus. The left label band spans from ``0`` to
    ``tu_mid - EDGE_EPS`` so tokens near the TransUnion seam are not
    misclassified.
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

    # Order the tokens left→right and extract midpoints
    mids.sort(key=lambda kv: kv[0])
    m0, m1, m2 = mids[0][0], mids[1][0], mids[2][0]

    # Boundaries halfway between midpoints (non-overlapping bands)
    b1 = (m0 + m1) / 2.0
    b2 = (m1 + m2) / 2.0

    label_left = 0.0
    label_right = max(0.0, m0 - EDGE_EPS_LABEL)

    tu_left = label_right
    tu_right = b1

    xp_left = b1
    xp_right = b2

    eq_left = b2
    eq_right = float("inf")

    layout = TriadLayout(
        page=page,
        label_band=(label_left, label_right),
        tu_band=(tu_left, tu_right),
        xp_band=(xp_left, xp_right),
        eq_band=(eq_left, eq_right),
    )

    # Explicit bounds log for debugging seam placement and label width
    triad_log(
        "TRIAD_LAYOUT_BOUNDS label=[0, %.1f) tu=[%.1f, %.1f) xp=[%.1f, %.1f) eq=[%.1f, inf)",
        layout.label_band[1],
        layout.tu_band[0],
        layout.tu_band[1],
        layout.xp_band[0],
        layout.xp_band[1],
        layout.eq_band[0],
    )
    return layout


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

    Tokens are classified by comparing their midpoint against the precomputed
    band edges. Bands do not overlap, so simple range checks are sufficient and
    avoid priority ordering bugs.
    """
    x = mid_x(token)

    # Right-side tie-break: if x is exactly on a boundary, assign to the band on the right
    if layout.label_band[0] <= x < layout.label_band[1]:
        return "label"
    if layout.tu_band[0] <= x < layout.tu_band[1]:
        return "tu"
    if layout.xp_band[0] <= x < layout.xp_band[1]:
        return "xp"
    if layout.eq_band[0] <= x:
        return "eq"
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


