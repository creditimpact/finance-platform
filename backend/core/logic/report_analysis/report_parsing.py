"""Utilities for parsing credit report PDFs into text and sections."""

import logging
import re
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

logger = logging.getLogger(__name__)


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


def extract_pdf_page_texts(pdf_path: str | Path, max_chars: int = 20000) -> list[str]:
    """Return a list of raw page texts from ``pdf_path``.

    Each page is truncated to ``max_chars`` characters to avoid excessive
    memory usage. Missing dependencies or extraction errors result in an empty
    list.
    """
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - fitz missing
        print(f"[WARN] PyMuPDF unavailable: {exc}")
        return []

    try:
        with fitz.open(pdf_path) as doc:
            pages = []
            for page in doc:
                txt = page.get_text()
                if len(txt) > max_chars:
                    txt = txt[:max_chars] + "\n[TRUNCATED]"
                pages.append(txt)
    except Exception as exc:  # pragma: no cover - runtime PDF issues
        print(f"[WARN] Failed to extract text from {pdf_path}: {exc}")
        return []
    return pages


_PAYMENT_MARK_RE = re.compile(r"payment\s*status", re.I)
_CREDITOR_MARK_RE = re.compile(r"creditor\s*remarks?", re.I)
_ACCOUNT_MARK_RE = re.compile(r"account\s*status|account\s*description", re.I)


def scan_page_markers(page_texts: Sequence[str]) -> dict[str, Any]:
    """Scan ``page_texts`` for marker strings and return a summary dict."""
    pages_payment_status: list[int] = []
    pages_creditor_remarks: list[int] = []
    pages_account_status: list[int] = []
    for idx, text in enumerate(page_texts, start=1):
        if _PAYMENT_MARK_RE.search(text):
            pages_payment_status.append(idx)
        if _CREDITOR_MARK_RE.search(text):
            pages_creditor_remarks.append(idx)
        if _ACCOUNT_MARK_RE.search(text):
            pages_account_status.append(idx)
    return {
        "has_payment_status": bool(pages_payment_status),
        "has_creditor_remarks": bool(pages_creditor_remarks),
        "has_account_status": bool(pages_account_status),
        "pages_payment_status": pages_payment_status,
        "pages_creditor_remarks": pages_creditor_remarks,
        "pages_account_status": pages_account_status,
    }


from backend.core.logic.utils.names_normalization import (  # noqa: E402
    normalize_bureau_name,
    normalize_creditor_name,
)
from backend.core.logic.utils.text_parsing import extract_account_blocks  # noqa: E402
from backend.core.models.bureau import BureauAccount  # noqa: E402

_PAYMENT_LABEL_RE = re.compile(r"payment\s*status[:]?", re.I)
_REMARKS_LABEL_RE = re.compile(r"creditor\s*remarks?[:]?")
_ACCOUNT_STATUS_LABEL_RE = re.compile(
    r"account\s*status|account\s*description[:]?", re.I
)


def extract_three_column_fields(
    pdf_path: str | Path,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    """Extract key rows split into three bureau columns using x-coordinates.

    Parameters
    ----------
    pdf_path:
        Path to the SmartCredit PDF report.

    Returns
    -------
    tuple
        Maps of payment statuses, remarks and account status/description per
        account along with raw line text for each category.
    """

    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - fitz missing
        logger.warning("PyMuPDF unavailable: %s", exc)
        return {}, {}, {}, {}, {}, {}

    payment_map: dict[str, dict[str, str]] = {}
    remarks_map: dict[str, dict[str, str]] = {}
    status_map: dict[str, dict[str, str]] = {}
    payment_raw: dict[str, str] = {}
    remarks_raw: dict[str, str] = {}
    status_raw: dict[str, str] = {}

    def _compute_ranges(spans, page_width: float) -> dict[str, tuple[float, float]]:
        positions: dict[str, float] = {}
        for sp in spans:
            low = sp.get("text", "").lower()
            x0 = sp.get("bbox", [0, 0, 0, 0])[0]
            if "transunion" in low:
                positions["TransUnion"] = x0
            elif "experian" in low:
                positions["Experian"] = x0
            elif "equifax" in low:
                positions["Equifax"] = x0
        if len(positions) != 3:
            return {}
        ordered = sorted(positions.items(), key=lambda x: x[1])
        bounds: dict[str, tuple[float, float]] = {}
        for idx, (name, pos) in enumerate(ordered):
            left = 0.0 if idx == 0 else (ordered[idx - 1][1] + pos) / 2
            right = (
                page_width
                if idx == len(ordered) - 1
                else (pos + ordered[idx + 1][1]) / 2
            )
            bounds[name] = (left, right)
        return bounds

    def _split_line(spans, ranges, label_re):
        raw = " ".join(sp.get("text", "") for sp in spans).strip()
        values: dict[str, str] = {k: "" for k in ranges}
        for sp in spans:
            text = sp.get("text", "")
            if label_re.search(text.lower()):
                continue
            x0 = sp.get("bbox", [0, 0, 0, 0])[0]
            for bureau, (xmin, xmax) in ranges.items():
                if xmin <= x0 < xmax:
                    values[bureau] += text + " "
                    break
        cleaned = {k: v.strip() for k, v in values.items() if v.strip()}
        return cleaned, raw

    with fitz.open(pdf_path) as doc:  # pragma: no cover - requires fitz
        for page in doc:
            width = float(page.rect.width)
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                lines = block.get("lines", [])
                if not lines:
                    continue
                heading_line = lines[0]
                heading_text = " ".join(
                    sp.get("text", "") for sp in heading_line.get("spans", [])
                ).strip()
                acc_norm = normalize_creditor_name(heading_text)
                ranges: dict[str, tuple[float, float]] | None = None
                for line in lines[1:]:
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text_line = " ".join(sp.get("text", "") for sp in spans).strip()
                    low = text_line.lower()
                    if ranges is None and all(
                        k in low for k in ("transunion", "experian", "equifax")
                    ):
                        ranges = _compute_ranges(spans, width)
                        continue
                    if not ranges:
                        continue
                    if _PAYMENT_LABEL_RE.search(low):
                        vals, raw = _split_line(spans, ranges, _PAYMENT_LABEL_RE)
                        if vals:
                            payment_map[acc_norm] = vals
                            payment_raw[acc_norm] = raw
                            for bureau, val in vals.items():
                                logger.info(
                                    "col_extract payment_status %s %s",
                                    bureau,
                                    val[:120],
                                )
                        continue
                    if _REMARKS_LABEL_RE.search(low):
                        vals, raw = _split_line(spans, ranges, _REMARKS_LABEL_RE)
                        if vals:
                            remarks_map[acc_norm] = vals
                            remarks_raw[acc_norm] = raw
                            for bureau, val in vals.items():
                                logger.info(
                                    "col_extract remarks %s %s", bureau, val[:120]
                                )
                        continue
                    if _ACCOUNT_STATUS_LABEL_RE.search(low):
                        vals, raw = _split_line(spans, ranges, _ACCOUNT_STATUS_LABEL_RE)
                        if vals:
                            status_map[acc_norm] = vals
                            status_raw[acc_norm] = raw
                            for bureau, val in vals.items():
                                logger.info(
                                    "col_extract account_status %s %s",
                                    bureau,
                                    val[:120],
                                )

    return payment_map, remarks_map, status_map, payment_raw, remarks_raw, status_raw


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

# ---------------------------------------------------------------------------
# Payment status parsing
# ---------------------------------------------------------------------------

# Account number extraction
ACCOUNT_NUMBER_ROW_RE = re.compile(
    r"(?:Account\s*(?:#|Number|No\.?)|Acct\s*(?:#|No\.?))\s*:?\s*(?P<tu>.+?)\s{2,}(?P<ex>.+?)\s{2,}(?P<eq>.+?)(?:\n|$)",
    re.I | re.S,
)
ACCOUNT_NUMBER_LINE_RE = re.compile(
    r"(?:Account\s*(?:#|Number|No\.?)|Acct\s*(?:#|No\.?))\s*:?\s*(.+)",
    re.I,
)


def _normalize_account_number(value: str) -> str | None:
    """Return a cleaned ``value`` if it contains at least one digit.

    The normalization removes whitespace and dashes while preserving any mask
    characters such as ``*``. If no digits are present the function returns
    ``None`` so callers can skip storing meaningless placeholders like
    ``"t disputed"``.
    """

    value = value.strip()
    if not re.search(r"\d", value):
        return None
    return re.sub(r"[\s-]", "", value)


def extract_payment_statuses(
    text: str,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Extract ``Payment Status`` lines for each bureau section.

    The function detects the bureau column boundaries from the header row in
    each account block and then slices the single ``Payment Status`` line into
    individual bureau values. The extracted values are normalized to lowercase
    with collapsed internal whitespace. A raw fallback of the right-hand side
    of the ``Payment Status`` line is also returned.

    Parameters
    ----------
    text:
        Raw text from the SmartCredit report.

    Returns
    -------
    tuple[dict[str, dict[str, str]], dict[str, str]]
        Two mappings: ``payment_statuses_by_heading`` and
        ``payment_status_raw_by_heading``.
    """

    def _normalize_val(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip()).lower()

    def _find_boundaries(lines: list[str]) -> list[str] | None:
        """Return bureau order based on the header row positions."""
        for line in lines:
            low = line.lower()
            if all(k in low for k in ("transunion", "experian", "equifax")):
                positions = {
                    "Transunion": low.index("transunion"),
                    "Experian": low.index("experian"),
                    "Equifax": low.index("equifax"),
                }
                ordered = sorted(positions.items(), key=lambda x: x[1])
                return [name for name, _ in ordered]
        positions: dict[str, int] = {}
        for line in lines:
            low = line.lower()
            if "transunion" in low and "Transunion" not in positions:
                positions["Transunion"] = low.index("transunion")
            if "experian" in low and "Experian" not in positions:
                positions["Experian"] = low.index("experian")
            if "equifax" in low and "Equifax" not in positions:
                positions["Equifax"] = low.index("equifax")
        if len(positions) == 3:
            ordered = sorted(positions.items(), key=lambda x: x[1])
            return [name for name, _ in ordered]
        return None

    statuses: dict[str, dict[str, str]] = {}
    raw_map: dict[str, str] = {}
    for block in extract_account_blocks(text):
        if not block:
            continue
        heading = block[0].strip()
        acc_norm = normalize_creditor_name(heading)

        bureau_order = _find_boundaries(block[1:])
        ps_line: str | None = None
        for line in block[1:]:
            if re.search(r"payment\s*status", line, re.I):
                ps_line = line
                break
        if ps_line:
            raw_match = re.search(r"Payment\s*Status\s*:?(.*)", ps_line, re.I)
            if raw_match:
                rhs = raw_match.group(1).strip()
                raw_map[acc_norm] = rhs
                if bureau_order:
                    parts = re.split(r"\s{2,}", rhs)
                    parts += ["", "", ""]
                    vals = {}
                    for bureau, part in zip(bureau_order, parts):
                        norm_val = _normalize_val(part)
                        if norm_val:
                            vals[bureau] = norm_val
                    if vals:
                        statuses[acc_norm] = vals

        # Additional per-bureau lines may specify payment status individually
        current_bureau: str | None = None
        for line in block[1:]:
            clean = line.strip()
            if (
                sum(
                    1
                    for b in ("TransUnion", "Experian", "Equifax")
                    if re.search(b, clean, re.I)
                )
                > 1
            ):
                current_bureau = None
                continue
            bureau_match = re.match(r"(TransUnion|Experian|Equifax)\b", clean, re.I)
            if bureau_match:
                current_bureau = normalize_bureau_name(bureau_match.group(1)).title()
                ps_inline = PAYMENT_STATUS_RE.search(clean)
                if ps_inline:
                    statuses.setdefault(acc_norm, {})[current_bureau] = _normalize_val(
                        ps_inline.group(1)
                    )
                continue

            if current_bureau and not re.search(r"payment\s*status", clean, re.I):
                ps = PAYMENT_STATUS_RE.match(clean)
                if ps:
                    statuses.setdefault(acc_norm, {})[current_bureau] = _normalize_val(
                        ps.group(1)
                    )

    return statuses, raw_map


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
            tu = _normalize_account_number(row.group("tu"))
            ex = _normalize_account_number(row.group("ex"))
            eq = _normalize_account_number(row.group("eq"))
            if tu:
                numbers.setdefault(acc_norm, {})[
                    normalize_bureau_name("TransUnion")
                ] = tu
            if ex:
                numbers.setdefault(acc_norm, {})[normalize_bureau_name("Experian")] = ex
            if eq:
                numbers.setdefault(acc_norm, {})[normalize_bureau_name("Equifax")] = eq

        current_bureau: str | None = None
        for line in block[1:]:
            clean = line.strip()
            bureau_match = re.match(r"(TransUnion|Experian|Equifax)\b", clean, re.I)
            if bureau_match:
                current_bureau = normalize_bureau_name(bureau_match.group(1))
                # Bureau line itself might contain the account number
                inline = ACCOUNT_NUMBER_LINE_RE.search(clean)
                if inline:
                    value = _normalize_account_number(inline.group(1))
                    if value:
                        numbers.setdefault(acc_norm, {})[current_bureau] = value
                continue

            if current_bureau:
                m = ACCOUNT_NUMBER_LINE_RE.match(clean)
                if m:
                    value = _normalize_account_number(m.group(1))
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
                    remarks.setdefault(acc_norm, {})[current_bureau] = rem_inline.group(
                        1
                    ).strip()
                continue

            if current_bureau:
                rem = CREDITOR_REMARKS_RE.match(clean)
                if rem:
                    remarks.setdefault(acc_norm, {})[current_bureau] = rem.group(
                        1
                    ).strip()

    return remarks
