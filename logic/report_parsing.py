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


from models.bureau import BureauAccount  # noqa: E402


def bureau_data_from_dict(data: dict) -> dict[str, list[BureauAccount]]:
    """Convert raw bureau ``data`` to typed ``BureauAccount`` objects.

    Parameters
    ----------
    data:
        Mapping of section name to list of account dictionaries.

    Returns
    -------
    dict[str, list[BureauAccount]]
        Mapping with the same keys but ``BureauAccount`` instances as values.
    """
    result: dict[str, list[BureauAccount]] = {}
    for section, items in data.items():
        if isinstance(items, list):
            result[section] = [BureauAccount.from_dict(it) for it in items]
    return result
