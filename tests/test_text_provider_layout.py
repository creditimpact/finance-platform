import os
from pathlib import Path

import pytest

from backend.core.text.text_provider import load_text_with_layout


def _make_pdf_with_headers(tmp_path: Path) -> str:
    # Create a simple one-page PDF with three headers and some values under each
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "three_cols.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    # Headers positions (x, y from left, bottom) — spread across the page
    headers = [
        (72, height - 72, "Transunion"),
        (250, height - 72, "Experian"),
        (430, height - 72, "Equifax"),
    ]
    for x, y, txt in headers:
        c.drawString(x, y, txt)

    # Row values under each column (roughly aligned by x)
    rows = [
        (72, height - 100, "****1111"), (250, height - 100, "****2222"), (430, height - 100, "****3333"),
        (72, height - 120, "$100"), (250, height - 120, "$200"), (430, height - 120, "$300"),
    ]
    for x, y, txt in rows:
        c.drawString(x, y, txt)

    c.showPage()
    c.save()
    return str(pdf_path)


def test_layout_api_shape(monkeypatch, tmp_path: Path):
    # Simulate extractor returning raw tokens without col; rely on post-pass to fill cols
    def fake_extract(pdf_path: str):
        # Construct a fake page with headers and values laid out in 3 columns
        tokens = []
        # headers
        tokens.append({"text": "Transunion", "x0": 50, "y0": 50, "x1": 150, "y1": 65, "line": 0, "col": None})
        tokens.append({"text": "Experian", "x0": 250, "y0": 50, "x1": 330, "y1": 65, "line": 0, "col": None})
        tokens.append({"text": "Equifax", "x0": 450, "y0": 50, "x1": 520, "y1": 65, "line": 0, "col": None})
        # row 1 values under each
        tokens.append({"text": "****1111", "x0": 60, "y0": 90, "x1": 120, "y1": 105, "line": 0, "col": None})
        tokens.append({"text": "****2222", "x0": 260, "y0": 90, "x1": 320, "y1": 105, "line": 0, "col": None})
        tokens.append({"text": "****3333", "x0": 460, "y0": 90, "x1": 520, "y1": 105, "line": 0, "col": None})
        page = {"number": 1, "width": 600, "height": 800, "tokens": tokens}
        return [page], "Transunion Experian Equifax\n..."

    import backend.core.text.text_provider as tp
    monkeypatch.setattr(tp, "_extract_pdf_tokens", fake_extract)

    out = load_text_with_layout("dummy.pdf")
    assert isinstance(out, dict) and "pages" in out and "full_text" in out
    pages = out["pages"]
    assert isinstance(pages, list) and len(pages) >= 1
    page0 = pages[0]
    assert set(["number", "width", "height", "tokens"]).issubset(page0.keys())
    tokens = page0["tokens"]
    assert isinstance(tokens, list) and len(tokens) > 0
    # Ensure required fields exist and col is restricted
    for t in tokens[:20]:
        assert set(["text", "x0", "y0", "x1", "y1", "line", "col"]).issubset(t.keys())
        assert t["col"] in (0, 1, 2, None)


def test_column_assignment_with_headers(monkeypatch, tmp_path: Path):
    # Reuse the same fake extractor and verify column mapping
    def fake_extract(pdf_path: str):
        tokens = []
        tokens.append({"text": "Transunion", "x0": 50, "y0": 50, "x1": 150, "y1": 65, "line": 0, "col": None})
        tokens.append({"text": "Experian", "x0": 250, "y0": 50, "x1": 330, "y1": 65, "line": 0, "col": None})
        tokens.append({"text": "Equifax", "x0": 450, "y0": 50, "x1": 520, "y1": 65, "line": 0, "col": None})
        tokens.append({"text": "****1111", "x0": 60, "y0": 90, "x1": 120, "y1": 105, "line": 0, "col": None})
        tokens.append({"text": "****2222", "x0": 260, "y0": 90, "x1": 320, "y1": 105, "line": 0, "col": None})
        tokens.append({"text": "****3333", "x0": 460, "y0": 90, "x1": 520, "y1": 105, "line": 0, "col": None})
        page = {"number": 1, "width": 600, "height": 800, "tokens": tokens}
        return [page], ""

    import backend.core.text.text_provider as tp
    monkeypatch.setattr(tp, "_extract_pdf_tokens", fake_extract)

    out = load_text_with_layout("dummy.pdf")
    tokens = out["pages"][0]["tokens"]
    # Find sample tokens per column by matching the values we drew
    col0 = [t for t in tokens if t["text"].endswith("1111")]
    col1 = [t for t in tokens if t["text"].endswith("2222")]
    col2 = [t for t in tokens if t["text"].endswith("3333")]
    assert col0 and all(t["col"] == 0 for t in col0)
    assert col1 and all(t["col"] == 1 for t in col1)
    assert col2 and all(t["col"] == 2 for t in col2)
