"""Utilities for parsing credit report PDFs into text and sections."""

from pathlib import Path
import re
from typing import Any, Mapping, cast


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
    from backend.core.logic.utils.pdf_ops import extract_pdf_text_safe

    return cast(str, extract_pdf_text_safe(Path(pdf_path), max_chars=150000))


from backend.core.models.bureau import BureauAccount  # noqa: E402
from backend.core.logic.utils.text_parsing import extract_account_blocks
from backend.core.logic.utils.names_normalization import (
    normalize_bureau_name,
    normalize_creditor_name,
)


def bureau_data_from_dict(
    data: Mapping[str, list[dict[str, Any]]],
) -> Mapping[str, list[BureauAccount]]:
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


PAYMENT_STATUS_RE = re.compile(r"payment status:\s*(.+)", re.I)
CREDITOR_REMARKS_RE = re.compile(r"creditor remarks:\s*(.+)", re.I)

PAYMENT_STATUS_ROW_RE = re.compile(
    r"Payment\s*Status\s*:?\s*(?P<tu>.+?)\s{2,}(?P<ex>.+?)\s{2,}(?P<eq>.+?)(?:\n|$)",
    re.I | re.S,
)

# Account number extraction
ACCOUNT_NUMBER_ROW_RE = re.compile(
    r"(?:Account\s*(?:#|Number|No\.?)|Acct\s*(?:#|No\.?))\s*:?\s*(?P<tu>.+?)\s{2,}(?P<ex>.+?)\s{2,}(?P<eq>.+?)(?:\n|$)",
    re.I | re.S,
)
ACCOUNT_NUMBER_LINE_RE = re.compile(
    r"(?:Account\s*(?:#|Number|No\.?)|Acct\s*(?:#|No\.?))\s*:?\s*(.+)",
    re.I,
)


def extract_payment_statuses(text: str) -> dict[str, dict[str, str]]:
    """Extract ``Payment Status`` lines for each bureau section.

    Parameters
    ----------
    text:
        Raw text from the SmartCredit report.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of normalized account names to a mapping of
        ``bureau -> payment status`` strings.
    """

    statuses: dict[str, dict[str, str]] = {}
    for block in extract_account_blocks(text):
        if not block:
            continue
        heading = block[0].strip()
        acc_norm = normalize_creditor_name(heading)

        block_text = "\n".join(block[1:])
        row = PAYMENT_STATUS_ROW_RE.search(block_text)
        if row:
            statuses.setdefault(acc_norm, {})[
                normalize_bureau_name("TransUnion")
            ] = row.group("tu").strip()
            statuses.setdefault(acc_norm, {})[
                normalize_bureau_name("Experian")
            ] = row.group("ex").strip()
            statuses.setdefault(acc_norm, {})[
                normalize_bureau_name("Equifax")
            ] = row.group("eq").strip()

        current_bureau: str | None = None
        for line in block[1:]:
            clean = line.strip()
            bureau_match = re.match(r"(TransUnion|Experian|Equifax)\b", clean, re.I)
            if bureau_match:
                current_bureau = normalize_bureau_name(bureau_match.group(1))
                # If the bureau line itself contains a payment status
                ps_inline = PAYMENT_STATUS_RE.search(clean)
                if ps_inline:
                    statuses.setdefault(acc_norm, {})[current_bureau] = ps_inline.group(1).strip()
                continue

            if current_bureau:
                ps = PAYMENT_STATUS_RE.match(clean)
                if ps:
                    statuses.setdefault(acc_norm, {})[current_bureau] = ps.group(1).strip()

    return statuses


def extract_account_numbers(text: str) -> dict[str, dict[str, str]]:
    """Extract account numbers for each bureau section.

    Parameters
    ----------
    text:
        Raw text from the SmartCredit report.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of normalized account names to ``bureau -> account_number``.
    """

    numbers: dict[str, dict[str, str]] = {}
    for block in extract_account_blocks(text):
        if not block:
            continue
        heading = block[0].strip()
        acc_norm = normalize_creditor_name(heading)

        block_text = "\n".join(block[1:])
        row = ACCOUNT_NUMBER_ROW_RE.search(block_text)
        if row:
            tu = row.group("tu").strip()
            ex = row.group("ex").strip()
            eq = row.group("eq").strip()
            if tu:
                numbers.setdefault(acc_norm, {})[
                    normalize_bureau_name("TransUnion")
                ] = tu
            if ex:
                numbers.setdefault(acc_norm, {})[
                    normalize_bureau_name("Experian")
                ] = ex
            if eq:
                numbers.setdefault(acc_norm, {})[
                    normalize_bureau_name("Equifax")
                ] = eq

        current_bureau: str | None = None
        for line in block[1:]:
            clean = line.strip()
            bureau_match = re.match(r"(TransUnion|Experian|Equifax)\b", clean, re.I)
            if bureau_match:
                current_bureau = normalize_bureau_name(bureau_match.group(1))
                # Bureau line itself might contain the account number
                inline = ACCOUNT_NUMBER_LINE_RE.search(clean)
                if inline:
                    value = inline.group(1).strip()
                    if value:
                        numbers.setdefault(acc_norm, {})[current_bureau] = value
                continue

            if current_bureau:
                m = ACCOUNT_NUMBER_LINE_RE.match(clean)
                if m:
                    value = m.group(1).strip()
                    if value:
                        numbers.setdefault(acc_norm, {})[current_bureau] = value

    return numbers


def extract_creditor_remarks(text: str) -> dict[str, dict[str, str]]:
    """Extract ``Creditor Remarks`` lines for each bureau section.

    Parameters
    ----------
    text:
        Raw text from the SmartCredit report.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of normalized account names to a mapping of
        ``bureau -> remarks`` strings.
    """

    remarks: dict[str, dict[str, str]] = {}
    for block in extract_account_blocks(text):
        if not block:
            continue
        heading = block[0].strip()
        acc_norm = normalize_creditor_name(heading)
        current_bureau: str | None = None
        for line in block[1:]:
            clean = line.strip()
            bureau_match = re.match(r"(TransUnion|Experian|Equifax)\b", clean, re.I)
            if bureau_match:
                current_bureau = bureau_match.group(1).title()
                # If the bureau line itself contains remarks
                rem_inline = CREDITOR_REMARKS_RE.search(clean)
                if rem_inline:
                    remarks.setdefault(acc_norm, {})[current_bureau] = rem_inline.group(1).strip()
                continue

            if current_bureau:
                rem = CREDITOR_REMARKS_RE.match(clean)
                if rem:
                    remarks.setdefault(acc_norm, {})[current_bureau] = rem.group(1).strip()

    return remarks
