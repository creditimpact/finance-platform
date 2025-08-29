from typing import List, Mapping

try:  # Allow import in environments without PyMuPDF binaries
    import fitz  # type: ignore  # PyMuPDF
except Exception:  # pragma: no cover - test environments may lack PyMuPDF
    fitz = None  # type: ignore


def extract_text_per_page(pdf_path: str) -> List[str]:
    """Return a list of page texts (one string per page).

    Never returns ``None``; empty string if no text.
    """

    if fitz is None:  # pragma: no cover - surfaced only if not monkeypatched
        raise ImportError("PyMuPDF (fitz) is required for PDF text extraction")

    texts: List[str] = []
    with fitz.open(pdf_path) as doc:  # thin wrapper
        for page in doc:
            texts.append(page.get_text("text") or "")
    return texts


def char_count(s: str) -> int:
    """Return the length of ``s`` treating ``None`` as empty string."""

    return len(s or "")


def merge_text_with_ocr(
    page_texts: List[str], ocr_texts: Mapping[int, str]
) -> List[str]:
    """Merge per-page OCR results into ``page_texts``.

    ``ocr_texts`` maps a 0-indexed page number to its OCR extracted text. Any
    non-empty OCR result replaces the corresponding entry in ``page_texts``.
    Pages without an OCR result remain unchanged.
    """

    merged = list(page_texts)
    for idx, txt in ocr_texts.items():
        if 0 <= idx < len(merged) and txt:
            merged[idx] = txt
    return merged
