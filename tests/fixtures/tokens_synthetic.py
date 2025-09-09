from __future__ import annotations

from typing import List, Dict


def three_bureau_tokens() -> List[Dict]:
    """Return a synthetic set of tokens with headers, labels and values across 3 columns.

    Coordinates are arbitrary but consistent. `col` is assigned per token.
    """
    toks: List[Dict] = []
    # Headers
    toks += [
        {"text": "Transunion", "x0": 50, "y0": 80, "x1": 150, "y1": 95, "line": 1, "col": 0},
        {"text": "Experian", "x0": 250, "y0": 80, "x1": 330, "y1": 95, "line": 1, "col": 1},
        {"text": "Equifax", "x0": 450, "y0": 80, "x1": 520, "y1": 95, "line": 1, "col": 2},
    ]
    # Labels
    toks += [
        {"text": "Account #", "x0": 30, "y0": 100, "x1": 120, "y1": 112, "line": 2, "col": None},
        {"text": "High Balance:", "x0": 30, "y0": 120, "x1": 150, "y1": 132, "line": 3, "col": None},
        {"text": "Account Type:", "x0": 30, "y0": 140, "x1": 140, "y1": 152, "line": 4, "col": None},
    ]
    # Values
    toks += [
        {"text": "****1111", "x0": 140, "y0": 100, "x1": 200, "y1": 112, "line": 2, "col": 0},
        {"text": "****2222", "x0": 260, "y0": 100, "x1": 320, "y1": 112, "line": 2, "col": 1},
        {"text": "****3333", "x0": 460, "y0": 100, "x1": 520, "y1": 112, "line": 2, "col": 2},
        {"text": "$1,000", "x0": 140, "y0": 120, "x1": 200, "y1": 132, "line": 3, "col": 0},
        {"text": "$2,000", "x0": 260, "y0": 120, "x1": 320, "y1": 132, "line": 3, "col": 1},
        {"text": "$3,000", "x0": 460, "y0": 120, "x1": 520, "y1": 132, "line": 3, "col": 2},
        {"text": "Conventional", "x0": 460, "y0": 140, "x1": 540, "y1": 152, "line": 4, "col": 2},
        {"text": "real", "x0": 540, "y0": 140, "x1": 580, "y1": 152, "line": 4, "col": 2},
        {"text": "estate", "x0": 580, "y0": 140, "x1": 640, "y1": 152, "line": 4, "col": 2},
        {"text": "mortgage", "x0": 460, "y0": 150, "x1": 540, "y1": 162, "line": 5, "col": 2},
        {"text": "Closed", "x0": 140, "y0": 140, "x1": 200, "y1": 152, "line": 4, "col": 0},
        {"text": "Paid", "x0": 260, "y0": 140, "x1": 300, "y1": 152, "line": 4, "col": 1},
    ]
    return toks

