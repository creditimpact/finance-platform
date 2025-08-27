from typing import List

import fitz  # PyMuPDF


def extract_text_per_page(pdf_path: str) -> List[str]:
    """Return a list of page texts (one string per page).

    Never returns ``None``; empty string if no text.
    """

    texts: List[str] = []
    with fitz.open(pdf_path) as doc:  # pragma: no cover - thin wrapper
        for page in doc:
            texts.append(page.get_text("text") or "")
    return texts


def char_count(s: str) -> int:
    """Return the length of ``s`` treating ``None`` as empty string."""

    return len(s or "")

