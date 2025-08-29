from __future__ import annotations


def extract_with_plumber(pdf_path: str) -> str:
    """Extract text using pdfplumber, concatenating all pages.

    Returns an empty string on failure; callers may choose to fallback.
    """

    try:
        import pdfplumber  # type: ignore
    except Exception:
        return ""

    parts: list[str] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:  # type: ignore
            for p in pdf.pages:
                parts.append(p.extract_text() or "")
    except Exception:
        return "\n".join(parts)
    return "\n".join(parts)


def extract_with_fitz(pdf_path: str) -> str:
    """Extract text using PyMuPDF (fitz) with layout-aware reflow."""

    try:
        import fitz  # type: ignore  # PyMuPDF
    except Exception:
        return ""

    def reflow_text(doc):  # type: ignore
        lines_all: list[str] = []
        try:
            for page in doc:
                try:
                    p = page.get_text("rawdict")
                except Exception:
                    # Fallback to basic text if rawdict fails
                    lines = (page.get_text("text") or "").splitlines()
                    lines_all.extend(lines)
                    continue

                # Collect spans with (y, x, text)
                spans: list[tuple[float, float, str]] = []
                for block in p.get("blocks", []) or []:
                    for line in block.get("lines", []) or []:
                        for span in line.get("spans", []) or []:
                            t = span.get("text", "")
                            if not t:
                                continue
                            x0, y0, _, _ = span.get("bbox", [0, 0, 0, 0])
                            spans.append((float(y0), float(x0), t))

                # Group by y with tolerance
                spans.sort(key=lambda a: (a[0], a[1]))
                grouped: list[list[tuple[float, float, str]]] = []
                y_tol = 2.0
                for y, x, t in spans:
                    if not grouped:
                        grouped.append([(y, x, t)])
                        continue
                    if abs(grouped[-1][0][0] - y) <= y_tol:
                        grouped[-1].append((y, x, t))
                    else:
                        grouped.append([(y, x, t)])

                # Reconstruct lines within each group, sorted by x with gap spaces
                for grp in grouped:
                    grp.sort(key=lambda a: a[1])
                    pieces: list[str] = []
                    last_x: float | None = None
                    for _, x, t in grp:
                        if last_x is not None:
                            gap = x - last_x
                            if gap > 8:  # heuristic gap -> space
                                pieces.append(" ")
                        pieces.append(t)
                        last_x = x + max(0, len(t) * 2)  # approximate advance
                    line = "".join(pieces).strip()
                    if line:
                        # Ensure bureau headings are on separate lines
                        low = line.lower()
                        if any(b in low for b in ("transunion", "experian", "equifax")):
                            lines_all.append(line)
                        else:
                            lines_all.append(line)
                # Add explicit newline between pages
                lines_all.append("")
        except Exception:
            pass
        return "\n".join(lines_all).strip()

    try:
        with fitz.open(pdf_path) as doc:  # type: ignore
            return reflow_text(doc)
    except Exception:
        return ""


def extract_text(pdf_path: str, *, prefer_fitz: bool = True) -> str:
    """Return text extracted from ``pdf_path`` using the chosen backend.

    prefer_fitz=True generally yields better linearization for SmartCredit
    reports. When the preferred backend returns an empty string, the other
    backend is attempted as a best-effort fallback.
    """

    primary = extract_with_fitz if prefer_fitz else extract_with_plumber
    secondary = extract_with_plumber if prefer_fitz else extract_with_fitz
    text = primary(pdf_path)
    if not text:
        text = secondary(pdf_path)
    return text or ""
