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


from backend.core.logic.utils.names_normalization import normalize_bureau_name  # noqa: E402
from backend.core.logic.utils.norm import normalize_heading  # noqa: E402
from backend.core.logic.utils.text_parsing import extract_account_blocks  # noqa: E402
from backend.core.models.bureau import BureauAccount  # noqa: E402
from .constants import (
    BUREAUS,
    ACCOUNT_FIELD_SET,
    INQUIRY_FIELDS,
    PUBLIC_INFO_FIELDS,
)
from .normalize import to_number, to_iso_date  # noqa: E402

# Mapping of account detail labels to canonical keys. Each tuple contains the
# canonical key and a regex that matches variations of the label in the PDF
# table. The parser is case/space tolerant.
_DETAIL_LABELS: list[tuple[str, re.Pattern[str]]] = [
    ("account_number", re.compile(r"(?:account|acct)\s*(?:#|number|no\.?)", re.I)),
    ("high_balance", re.compile(r"high\s*balance", re.I)),
    ("last_verified", re.compile(r"last\s*verified", re.I)),
    ("date_of_last_activity", re.compile(r"date\s*of\s*last\s*activity", re.I)),
    ("date_reported", re.compile(r"date\s*reported", re.I)),
    ("date_opened", re.compile(r"date\s*opened", re.I)),
    ("balance_owed", re.compile(r"balance\s*owed", re.I)),
    ("closed_date", re.compile(r"closed\s*date|date\s*closed", re.I)),
    ("account_rating", re.compile(r"account\s*rating", re.I)),
    ("account_description", re.compile(r"account\s*description", re.I)),
    ("dispute_status", re.compile(r"dispute\s*status", re.I)),
    ("creditor_type", re.compile(r"creditor\s*type", re.I)),
    ("account_status", re.compile(r"account\s*status", re.I)),
    ("payment_status", re.compile(r"payment\s*status", re.I)),
    ("creditor_remarks", re.compile(r"creditor\s*remarks?", re.I)),
    ("payment_amount", re.compile(r"payment\s*amount", re.I)),
    ("last_payment", re.compile(r"last\s*payment", re.I)),
    ("term_length", re.compile(r"term\s*length", re.I)),
    ("past_due_amount", re.compile(r"past\s*due\s*amount", re.I)),
    ("account_type", re.compile(r"account\s*type", re.I)),
    ("payment_frequency", re.compile(r"payment\s*frequency", re.I)),
    ("credit_limit", re.compile(r"credit\s*limit", re.I)),
]

_MONEY_FIELDS = {
    "high_balance",
    "balance_owed",
    "credit_limit",
    "past_due_amount",
    "payment_amount",
}

_DATE_FIELDS = {
    "date_opened",
    "closed_date",
    "date_reported",
    "last_payment",
    "last_verified",
    "date_of_last_activity",
}


def _normalize_date(value: str) -> str | None:
    """Normalize various date formats to ``YYYY-MM`` or ``YYYY-MM-DD``."""

    value = value.strip()
    if not value:
        return None
    from datetime import datetime

    fmts_day = ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%b %d %Y", "%B %d %Y"]
    fmts_month = ["%m/%Y", "%Y-%m", "%b %Y", "%B %Y"]
    for fmt in fmts_day:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    for fmt in fmts_month:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m")
        except ValueError:
            pass
    return None


def _normalize_detail_value(key: str, value: str) -> tuple[Any | None, str | None]:
    """Return normalized value and raw string for ``key``."""

    raw = value.strip()
    if not raw:
        return None, None
    if key == "account_number":
        cleaned = re.sub(r"\s+", "", raw)
        if not re.search(r"\d", cleaned):
            return None, None
        return cleaned, raw
    if key in _MONEY_FIELDS:
        digits = re.sub(r"[^0-9]", "", raw)
        if not digits:
            return None, raw
        return int(digits), raw
    if key in _DATE_FIELDS:
        norm = _normalize_date(raw)
        return norm, raw
    return raw, raw


def extract_three_column_fields(
    pdf_path: str | Path,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, dict[str, Any]]],
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
        return {}, {}, {}, {}, {}, {}, {}

    payment_map: dict[str, dict[str, str]] = {}
    remarks_map: dict[str, dict[str, str]] = {}
    status_map: dict[str, dict[str, str]] = {}
    payment_raw: dict[str, str] = {}
    remarks_raw: dict[str, str] = {}
    status_raw: dict[str, str] = {}
    details_map: dict[str, dict[str, dict[str, Any]]] = {}

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
                acc_norm = normalize_heading(heading_text)
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
                    for key, label_re in _DETAIL_LABELS:
                        if label_re.search(low):
                            vals, raw = _split_line(spans, ranges, label_re)
                            if not vals:
                                break
                            for bureau, val in vals.items():
                                norm_val, raw_val = _normalize_detail_value(key, val)
                                if norm_val is None:
                                    continue
                                details_map.setdefault(acc_norm, {}).setdefault(bureau, {})[
                                    key
                                ] = norm_val
                                if raw_val and raw_val != norm_val:
                                    details_map[acc_norm][bureau][key + "_raw"] = raw_val
                                if key == "payment_status":
                                    payment_map.setdefault(acc_norm, {})[bureau] = (
                                        str(norm_val).lower()
                                        if isinstance(norm_val, str)
                                        else str(norm_val)
                                    )
                                if key == "creditor_remarks":
                                    remarks_map.setdefault(acc_norm, {})[bureau] = str(
                                        norm_val
                                    )
                                if key in {"account_status", "account_description"}:
                                    status_map.setdefault(acc_norm, {})[bureau] = str(
                                        norm_val
                                    )
                                logger.info(
                                    "col_extract %s %s %s",
                                    key,
                                    bureau,
                                    str(norm_val)[:120],
                                )
                            if key == "payment_status":
                                payment_raw[acc_norm] = raw
                            elif key == "creditor_remarks":
                                remarks_raw[acc_norm] = raw
                            elif key in {"account_status", "account_description"}:
                                status_raw[acc_norm] = raw
                            break

    return (
        payment_map,
        remarks_map,
        status_map,
        payment_raw,
        remarks_raw,
        status_raw,
        details_map,
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
        acc_norm = normalize_heading(heading)

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
        acc_norm = normalize_heading(heading)

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
        acc_norm = normalize_heading(heading)
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


# ---------------------------------------------------------------------------
# Bureau meta tables for accounts (raw.account_history.by_bureau)
# ---------------------------------------------------------------------------


def _ensure_paths(obj: dict, *path: str) -> dict:
    """Ensure nested dictionaries exist for ``path`` and return the final node."""

    cur: dict = obj
    for key in path:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    return cur


def _empty_bureau_map() -> dict[str, Any]:
    """Return a single-bureau map with the 25-field set all ``None``."""

    return {field: None for field in ACCOUNT_FIELD_SET}


def _to_num(s: str) -> Any | None:
    s = (s or "").strip()
    if s in {"--", "-", ""}:
        return None
    return to_number(s)


def _to_iso(s: str) -> Any | None:
    s = (s or "").strip()
    if s in {"--", "-", ""}:
        return None
    return to_iso_date(s)


BUREAU_LINE_RE = re.compile(
    r"^(Transunion|Experian|Equifax)\s+([0-9\*]+)\s+([\d,]+|0|--)\s+([-\d/]+|--)\s+([-\d/]+|--)\s+([-\d/]+|--)\s+([-\d/]+|--)\s+([\d,]+|0|--)\s+--?\s+(\w+)\s+(.*?)\s+(Bank|All Banks|National.*|.*?)\s+(.*?)\s+(Current|Late|.*?)(?:\s+--)?\s+(\d+|0|--)\s+([-\d/]+|--)\s+--?\s+(\d+|0|--)",
    re.I,
)


def parse_three_footer_lines(lines: list[str]) -> dict[str, dict[str, Any | None]]:
    """Parse the trailing three footer lines mapping to the bureaus.

    Each footer line corresponds to a single bureau in TU→EX→EQ order and may
    contain an account type, optional payment frequency and an optional credit
    limit ("--" or empty values are treated as ``None``).
    """

    out = {
        b: {"account_type": None, "payment_frequency": None, "credit_limit": None}
        for b in BUREAUS
    }

    # Identify the three lines immediately preceding the two-year history block
    try:
        hist_idx = next(
            i for i, ln in enumerate(lines) if "two-year payment history" in ln.lower()
        )
    except StopIteration:
        hist_idx = len(lines)

    candidates = [ln.strip() for ln in lines[max(0, hist_idx - 3) : hist_idx] if ln.strip()]
    if len(candidates) > 3:
        candidates = candidates[-3:]

    for line, bureau in zip(candidates, BUREAUS):
        parts = line.split()
        if not parts:
            continue
        # Credit limit is assumed to be the last token and numeric
        credit_limit = _to_num(parts[-1])
        if credit_limit is not None:
            parts = parts[:-1]
        else:
            credit_limit = None

        payment_frequency = None
        if parts:
            freq_cand = parts[-1].lower()
            if freq_cand in {
                "monthly",
                "weekly",
                "bi-weekly",
                "semi-monthly",
                "quarterly",
                "annually",
                "yearly",
                "--",
            }:
                if freq_cand != "--":
                    payment_frequency = freq_cand
                parts = parts[:-1]

        account_type = " ".join(parts).strip().lower() or None

        out[bureau]["account_type"] = account_type
        out[bureau]["payment_frequency"] = payment_frequency
        out[bureau]["credit_limit"] = credit_limit

    return out


def parse_two_year_history(lines: list[str]) -> dict[str, Any | None]:
    """Extract per-bureau two-year payment history strings."""

    out = {b: None for b in BUREAUS}

    try:
        start = next(i for i, ln in enumerate(lines) if "two-year payment history" in ln.lower())
        end = next(
            i
            for i, ln in enumerate(lines[start + 1 :], start + 1)
            if "days late -7 year history" in ln.lower()
        )
    except StopIteration:
        return out

    seg_lines = lines[start + 1 : end]
    current: str | None = None
    buffer: list[str] = []
    for ln in seg_lines:
        m = re.match(r"(Transunion|Experian|Equifax)\s*(.*)", ln, re.I)
        if m:
            if current and buffer:
                out[current] = " ".join(" ".join(buffer).split())
            current = m.group(1).lower()
            buffer = [m.group(2)]
        else:
            if current:
                buffer.append(ln)
    if current and buffer:
        out[current] = " ".join(" ".join(buffer).split())
    return out


def parse_seven_year_days_late(lines: list[str]) -> dict[str, Any | None]:
    """Sum 30/60/90 day late counts over seven years per bureau."""

    out = {b: None for b in BUREAUS}

    try:
        start = next(
            i for i, ln in enumerate(lines) if "days late -7 year history" in ln.lower()
        )
    except StopIteration:
        return out

    text = "\n".join(lines[start + 1 :])
    for b in BUREAUS:
        m = re.search(rf"{b}.*?30:(\d+)\s+60:(\d+)\s+90:(\d+)", text, re.I)
        if m:
            d30, d60, d90 = map(int, m.groups())
            out[b] = d30 + d60 + d90
    return out


def parse_account_block(block_lines: list[str]) -> dict[str, dict[str, Any | None]]:
    logger.debug("parse_account_block start lines=%d", len(block_lines))

    def _init_maps():
        return {
            b: {
                "account_number_display": None,
                "account_number_last4": None,
                "high_balance": None,
                "last_verified": None,
                "date_of_last_activity": None,
                "date_reported": None,
                "date_opened": None,
                "balance_owed": None,
                "closed_date": None,
                "account_rating": None,
                "account_description": None,
                "dispute_status": None,
                "creditor_type": None,
                "account_status": None,
                "payment_status": None,
                "creditor_remarks": None,
                "payment_amount": None,
                "last_payment": None,
                "term_length": None,
                "past_due_amount": None,
                "account_type": None,
                "payment_frequency": None,
                "credit_limit": None,
                "two_year_payment_history": None,
                "seven_year_days_late": None,
            }
            for b in BUREAUS
        }

    bureau_maps = _init_maps()
    raw_lines = [l.rstrip("\n") for l in block_lines if l.strip()]

    # --- Header-based column spans ---------------------------------------
    header_idx: int | None = None
    header_line: str | None = None
    for i, line in enumerate(raw_lines):
        low = line.lower()
        if "account #" in low and "high balance" in low:
            header_idx = i
            header_line = line
            break

    spans: list[tuple[str, int, int]] = []
    if header_line is not None:
        cols = [
            ("account_number_display", "account #"),
            ("high_balance", "high balance"),
            ("last_verified", "last verified"),
            ("date_of_last_activity", "date of last activity"),
            ("date_reported", "date reported"),
            ("date_opened", "date opened"),
            ("balance_owed", "balance owed"),
            ("closed_date", "closed date"),
            ("account_rating", "account rating"),
            ("account_description", "account description"),
            ("dispute_status", "dispute status"),
            ("creditor_type", "creditor type"),
            ("account_status", "account status"),
            ("payment_status", "payment status"),
            ("creditor_remarks", "creditor remarks"),
            ("payment_amount", "payment amount"),
            ("last_payment", "last payment"),
            ("term_length", "term length"),
            ("past_due_amount", "past due amount"),
        ]
        hlow = header_line.lower()
        pos_list: list[tuple[int, str]] = []
        for key, label in cols:
            idx = hlow.find(label)
            if idx >= 0:
                pos_list.append((idx, key))
        pos_list.sort()
        for idx, (start, key) in enumerate(pos_list):
            end = pos_list[idx + 1][0] if idx + 1 < len(pos_list) else len(header_line)
            spans.append((key, start, end))

    parsed = set()
    if spans and header_idx is not None:
        for line in raw_lines[header_idx + 1 :]:
            m = re.match(r"(Transunion|Experian|Equifax)\s+(.*)", line, re.I)
            if not m:
                continue
            bureau = m.group(1).lower()
            body = m.group(2)
            body = body.ljust(len(header_line))
            bm = bureau_maps[bureau]
            for key, start, end in spans:
                seg = body[start:end].strip()
                if seg in {"", "--"}:
                    val = None
                else:
                    if key == "account_number_display":
                        val = seg
                        digits = re.sub(r"\D", "", seg)
                        bm["account_number_last4"] = digits[-4:] if digits else None
                    elif key in {"high_balance", "balance_owed", "payment_amount", "past_due_amount"}:
                        val = _to_num(seg)
                    elif key in {
                        "last_verified",
                        "date_of_last_activity",
                        "date_reported",
                        "date_opened",
                        "last_payment",
                        "closed_date",
                    }:
                        val = _to_iso(seg)
                    else:
                        val = seg
                bm[key] = val
            parsed.add(bureau)

    # Fallback to legacy regex parsing if header spans failed
    if not parsed:
        joined = [" ".join(l.split()) for l in raw_lines]
        bureau_rows = [l for l in joined if re.match(r"^(Transunion|Experian|Equifax)\s", l, re.I)]
        for row in bureau_rows:
            m = BUREAU_LINE_RE.search(row)
            if not m:
                continue
            bname = m.group(1).lower()
            b = "transunion" if "trans" in bname else ("experian" if "exp" in bname else "equifax")
            masked = m.group(2)
            bm = bureau_maps[b]
            bm["account_number_display"] = masked
            bm["account_number_last4"] = (
                re.sub(r"\D", "", masked)[-4:] if re.search(r"\d", masked) else None
            )
            bm["high_balance"] = _to_num(m.group(3))
            bm["last_verified"] = _to_iso(m.group(4))
            bm["date_of_last_activity"] = _to_iso(m.group(5))
            bm["date_reported"] = _to_iso(m.group(6))
            bm["date_opened"] = _to_iso(m.group(7))
            bm["balance_owed"] = _to_num(m.group(8))
            bm["account_rating"] = m.group(9)
            bm["account_description"] = m.group(10)
            bm["creditor_type"] = m.group(11)
            bm["account_status"] = m.group(12)
            bm["payment_status"] = m.group(13)
            bm["payment_amount"] = _to_num(m.group(14))
            bm["last_payment"] = _to_iso(m.group(15))
            bm["past_due_amount"] = _to_num(m.group(16))

    # Footer triplet lines (account type / payment frequency / credit limit)
    footer = parse_three_footer_lines(raw_lines)
    for b in BUREAUS:
        for k in ("account_type", "payment_frequency", "credit_limit"):
            val = footer.get(b, {}).get(k)
            if val is not None:
                bureau_maps[b][k] = val

    hist2y = parse_two_year_history(raw_lines)
    sev7 = parse_seven_year_days_late(raw_lines)
    for b in BUREAUS:
        bm = bureau_maps[b]
        bm["two_year_payment_history"] = hist2y[b]
        bm["seven_year_days_late"] = sev7[b]
    return bureau_maps


def _find_bureau_entry(acc: Mapping[str, Any], bureau: str) -> Mapping[str, Any] | None:
    """Find matching entry in acc['bureaus'] for a bureau (accepts title/lower)."""
    items = acc.get("bureaus") or []
    if not isinstance(items, list):
        return None
    targets = {bureau, bureau.lower(), bureau.title()}
    for it in items:
        if not isinstance(it, Mapping):
            continue
        bname = it.get("bureau") or it.get("name")
        if isinstance(bname, str) and bname in targets:
            return it
    return None


def _fill_bureau_map_from_sources(
    acc: Mapping[str, Any],
    bureau: str,
    dst: dict[str, Any],
    account_block_lines: list[str] | None = None,
) -> None:
    """Fill a single bureau's 25-field map for an account.

    Source priority (highest to lowest):
    - acc.bureaus[]
    - acc.bureau_details[bureau]
    - acc.raw.account_history.by_bureau[bureau] (existing)

    Gentle normalization for numeric/date fields only when unambiguous.
    Also backfills account_number_display/last4 from top-level when missing.
    """

    # 1) bureaus[] entry
    src = _find_bureau_entry(acc, bureau)
    if isinstance(src, Mapping):
        # Map overlapping keys (identity plus known aliases)
        mapping: dict[str, str] = {f: f for f in ACCOUNT_FIELD_SET}
        mapping.update({
            "balance": "balance_owed",
            "last_reported": "date_reported",
            "date_reported": "date_reported",
            "reported_date": "date_reported",
            "remarks": "creditor_remarks",
            "status": "account_status",
            "rating": "account_rating",
        })
        for s_key, d_key in mapping.items():
            if dst.get(d_key) is None and src.get(s_key) not in (None, "", {}, []):
                val = src.get(s_key)
                if d_key in {"high_balance", "balance_owed", "credit_limit", "past_due_amount", "payment_amount"}:
                    dst[d_key] = to_number(val)
                elif d_key in {"date_opened", "date_reported", "closed_date", "last_verified", "last_payment", "date_of_last_activity"}:
                    dst[d_key] = to_iso_date(val)
                else:
                    dst[d_key] = val

    # 2) bureau_details[bureau]
    details = (acc.get("bureau_details") or {}).get(bureau)
    if not isinstance(details, Mapping):
        # Try normalized keys in case caller stored lowercased keys
        details = (acc.get("bureau_details") or {}).get(bureau.lower()) or (acc.get("bureau_details") or {}).get(bureau.title())
    if isinstance(details, Mapping):
        for key in ACCOUNT_FIELD_SET:
            if dst.get(key) is not None:
                continue
            if details.get(key) in (None, "", {}, []):
                continue
            val = details.get(key)
            if key in {"high_balance", "balance_owed", "credit_limit", "past_due_amount", "payment_amount"}:
                dst[key] = to_number(val)
            elif key in {"date_opened", "date_reported", "closed_date", "last_verified", "last_payment", "date_of_last_activity"}:
                dst[key] = to_iso_date(val)
            else:
                dst[key] = val

    # 3) Existing raw.by_bureau values as last resort
    try:
        existing = (
            acc.get("raw", {})
            .get("account_history", {})
            .get("by_bureau", {})
            .get(bureau, {})
        )
        if isinstance(existing, Mapping):
            for key in ACCOUNT_FIELD_SET:
                if dst.get(key) is None and existing.get(key) not in (None, "", {}, []):
                    dst[key] = existing.get(key)
    except Exception:
        pass

    # Backfill account number fields (top-level fallbacks)
    if dst.get("account_number_last4") in (None, ""):
        last4 = acc.get("account_number_last4") or acc.get("account_number")
        if isinstance(last4, str):
            digits = re.sub(r"\D", "", last4)
            dst["account_number_last4"] = digits[-4:] if digits else None
        elif isinstance(last4, (int, float)):
            s = str(int(last4))
            dst["account_number_last4"] = s[-4:] if s else None

    if dst.get("account_number_display") in (None, ""):
        disp = acc.get("account_number_raw") or acc.get("account_number_display") or acc.get("account_number")
        if disp not in (None, "", {}, []):
            dst["account_number_display"] = disp

    # --- BEFORE return, after merging from known sources ---
    missing_before = [k for k, v in dst.items() if v is None]
    logger.debug(
        "pre-parse gap: account=%s bureau=%s missing=%d top=%s",
        acc.get("normalized_name") or acc.get("name"),
        bureau,
        len(missing_before),
        ",".join(missing_before[:5]),
    )

    # If still missing keys and block lines were provided, backfill via parser
    if missing_before and account_block_lines:
        try:
            block_maps = parse_account_block(account_block_lines)
            bm = block_maps.get(bureau, {})
            for k in dst.keys():
                if dst[k] is None and k in bm:
                    dst[k] = bm[k]
        except Exception:
            logger.exception(
                "parse_account_block_failed account=%s bureau=%s",
                acc.get("normalized_name") or acc.get("name"),
                bureau,
            )

    filled = sum(1 for v in dst.values() if v is not None)
    logger.info(
        "parser_bureau_fill account=%s bureau=%s filled=%d/25",
        acc.get("normalized_name") or acc.get("name"),
        bureau,
        filled,
    )


def attach_bureau_meta_tables(sections: Mapping[str, Any]) -> None:
    """Attach per-bureau meta tables and supplemental RAW blocks."""

    accounts = sections.get("all_accounts") or []
    if not isinstance(accounts, list):
        return

    session_id = sections.get("session_id") or ""

    # Normalize report-level inquiries/public info once
    inq_src = sections.get("inquiries") or []
    norm_inqs: list[dict[str, Any]] = []
    if isinstance(inq_src, list):
        for inq in inq_src:
            if not isinstance(inq, Mapping):
                continue
            bureau = (
                normalize_bureau_name(inq.get("bureau")) if inq.get("bureau") else None
            )
            item = {
                "bureau": bureau.lower() if bureau else None,
                "subscriber": inq.get("subscriber") or inq.get("creditor_name") or inq.get("name"),
                "date": to_iso_date(inq.get("date")) if inq.get("date") else None,
                "type": inq.get("type"),
                "permissible_purpose": inq.get("permissible_purpose"),
                "remarks": inq.get("remarks"),
                "_provenance": inq.get("_provenance", {}),
            }
            for k in INQUIRY_FIELDS:
                item.setdefault(k, None)
            norm_inqs.append(item)

    pub_src = sections.get("public_information") or []
    norm_pub: list[dict[str, Any]] = []
    if isinstance(pub_src, list):
        for item in pub_src:
            if not isinstance(item, Mapping):
                continue
            bureau = (
                normalize_bureau_name(item.get("bureau")) if item.get("bureau") else None
            )
            date_val = item.get("date_filed") or item.get("date")
            pi = {
                "bureau": bureau.lower() if bureau else None,
                "item_type": item.get("item_type") or item.get("type"),
                "status": item.get("status"),
                "date_filed": to_iso_date(date_val) if date_val else None,
                "amount": to_number(item.get("amount")) if item.get("amount") else None,
                "remarks": item.get("remarks"),
                "_provenance": item.get("_provenance", {}),
            }
            for k in PUBLIC_INFO_FIELDS:
                pi.setdefault(k, None)
            norm_pub.append(pi)

    for acc in accounts:
        if not isinstance(acc, dict):
            continue
        by = _ensure_paths(acc, "raw", "account_history", "by_bureau")

        raw = acc.setdefault("raw", {})
        raw.setdefault("inquiries", {"items": []})
        raw.setdefault("public_information", {"items": []})

        if norm_inqs:
            if not raw.get("inquiries", {}).get("items"):
                raw["inquiries"]["items"] = norm_inqs
        elif inq_src:
            slug = acc.get("account_id") or acc.get("normalized_name") or acc.get("name") or ""
            logger.warning(
                "inquiries_detected_but_not_written session=%s account=%s",
                session_id,
                slug,
            )

        if norm_pub:
            if not raw.get("public_information", {}).get("items"):
                raw["public_information"]["items"] = norm_pub
        elif pub_src:
            slug = acc.get("account_id") or acc.get("normalized_name") or acc.get("name") or ""
            logger.warning(
                "public_info_detected_but_not_written session=%s account=%s",
                session_id,
                slug,
            )

        account_block_lines = (
            sections.get("blocks_by_account", {})
            .get(acc.get("normalized_name") or acc.get("name"))
        )

        for b in BUREAUS:
            dst = by.get(b)
            if not isinstance(dst, Mapping):
                dst = _empty_bureau_map()
            else:
                for field in ACCOUNT_FIELD_SET:
                    dst.setdefault(field, None)
            _fill_bureau_map_from_sources(acc, b, dst, account_block_lines)
            by[b] = dst

        tu = sum(1 for v in by.get("transunion", {}).values() if v is not None)
        ex = sum(1 for v in by.get("experian", {}).values() if v is not None)
        eq = sum(1 for v in by.get("equifax", {}).values() if v is not None)
        logger.info(
            "bureau_meta_coverage name=%s tu_missing=%d ex_missing=%d eq_missing=%d",
            acc.get("normalized_name") or acc.get("name"),
            25 - tu,
            25 - ex,
            25 - eq,
        )
