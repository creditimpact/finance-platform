"""Utilities for parsing credit report PDFs into text and sections."""

from pathlib import Path


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Return text extracted from *pdf_path* using a robust multi-engine approach.

    The heavy :mod:`fitz` dependency is imported lazily to avoid import-time
    side effects in modules that merely type-check or reference this function.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file to be parsed.

    Returns
    -------
    str
        The extracted text limited to a sensible character count to avoid
        excessive memory consumption.
    """
    from .utils.pdf_ops import extract_pdf_text_safe

    return extract_pdf_text_safe(Path(pdf_path), max_chars=150000)
