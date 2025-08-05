from pathlib import Path
from fpdf import FPDF
import pdfplumber
import fitz
import re

BUREAUS = ["Experian", "Equifax", "TransUnion"]

# Allow a few common variations when looking up bureaus
BUREAU_ALIASES = {
    "transunion": "TransUnion",
    "trans union": "TransUnion",
    "tu": "TransUnion",
    "experian": "Experian",
    "exp": "Experian",
    "ex": "Experian",
    "equifax": "Equifax",
    "eq": "Equifax",
    "efx": "Equifax",
}


def normalize_creditor_name(name: str) -> str:
    """Proxy to :func:`generate_goodwill_letters.normalize_creditor_name`."""
    from .generate_goodwill_letters import normalize_creditor_name as _norm

    return _norm(name)


def normalize_bureau_name(name: str | None) -> str:
    """Return canonical bureau name for various capitalizations/aliases."""
    if not name:
        return ""
    key = name.strip().lower()
    return BUREAU_ALIASES.get(key, name.title())


def safe_filename(name: str) -> str:
    """Return a filename-safe version of ``name``."""
    cleaned = name.strip().replace(" ", "_")
    return re.sub(r"[\\/:*?\"<>|]", "_", cleaned)


# -----------------------------------------
# Custom note handling helpers
# -----------------------------------------

HARDSHIP_RE = re.compile(
    r"(lost my job|job loss|layoff|medical|illness|hospital|covid|pandemic|family emergency|divorce|funeral|death in|financial hardship|hardship|sick)",
    re.I,
)


def is_general_hardship_note(text: str | None) -> bool:
    """Return True if the note appears to be a general hardship explanation."""
    if not text:
        return False
    return bool(HARDSHIP_RE.search(text.strip().lower()))


def analyze_custom_notes(custom_notes: dict, account_names: list[str]):
    """Separate account-specific notes from general hardship notes.

    Returns a tuple ``(specific_notes, general_notes)`` where ``specific_notes``
    is a mapping of normalized account names to notes.
    """

    normalized_accounts = {normalize_creditor_name(n) for n in account_names}
    specific: dict[str, str] = {}
    general: list[str] = []

    for key, note in (custom_notes or {}).items():
        if not note:
            continue
        key_norm = normalize_creditor_name(key)
        if key_norm in normalized_accounts and not is_general_hardship_note(note):
            specific[key_norm] = note.strip()
        else:
            general.append(str(note).strip())

    return specific, general


def get_client_address_lines(client_info: dict) -> list[str]:
    """Return client's mailing address lines.

    Priority order:
    1. ``client_info['address']``
    2. ``client_info['current_address']`` extracted from the credit report

    The returned list may contain one or two lines. When no address is found,
    an empty list is returned so the caller can render a placeholder line.
    """

    raw = (client_info.get("address") or client_info.get("current_address") or "").strip()
    if not raw:
        return []

    # Normalize separators to detect street vs city/state/zip parts
    raw = raw.replace("\r", "\n")
    parts = [p.strip() for p in re.split(r"\n|,", raw) if p.strip()]

    if len(parts) >= 2:
        line1 = parts[0]
        line2 = ", ".join(parts[1:])
        return [line1, line2]
    return [raw]


LATE_PATTERN = re.compile(
    r"(\d+\s*x\s*30|\d+\s*x\s*60|30[-\s]*day[s]?\s*late|60[-\s]*day[s]?\s*late|90[-\s]*day[s]?\s*late|late payment|past due)",
    re.I,
)

NO_LATE_PATTERN = re.compile(
    r"(no late payments|never late|never been late|no history of late)",
    re.I,
)

GENERIC_NAME_RE = re.compile(r"days?\s+late|payment\s+history|year\s+history", re.I)


def _has_late_flag(text: str) -> bool:
    """Return True when the text indicates late payments."""
    clean = str(text or "").lower()
    if NO_LATE_PATTERN.search(clean):
        return False
    if re.search(r"0\s*x\s*(30|60|90)", clean):
        return False
    return bool(LATE_PATTERN.search(clean))


_MONTHS = {
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}


def _is_calendar_line(line: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+|\d+", line.lower())
    has_month = False
    for t in tokens:
        if t.isdigit():
            continue
        if t in _MONTHS:
            has_month = True
            continue
        return False
    return has_month


def _potential_account_name(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if lower == "none reported" or _is_calendar_line(stripped):
        return False
    if re.fullmatch(r"(ok\b\s*){2,}", lower):
        return False
    tokens = lower.split()
    if len(tokens) >= 2 and len(set(tokens)) == 1 and len(tokens[0]) <= 3:
        return False
    if not re.fullmatch(r"[A-Z/&\-\s]+", stripped):
        return False
    return (
        len(stripped) >= 3
        and not re.match(r"(TransUnion|Experian|Equifax)\b", stripped, re.I)
        and re.search(r"[A-Z]", stripped)
        and not GENERIC_NAME_RE.search(stripped)
    )


def _parse_late_counts(segment: str) -> dict[str, int]:
    """Return late payment counts avoiding account number artifacts."""
    counts: dict[str, int] = {}
    pattern = re.compile(r"(?<!\d)(30|60|90)[\s-]*:?\s*(\d+)(?!\d)")
    for part in pattern.finditer(segment):
        num = int(part.group(2))
        if num > 12:
            print(f"[~] Ignoring unrealistic late count {num} for {part.group(1)}")
            continue
        counts[part.group(1)] = num
    return counts


ACCOUNT_FIELD_RE = re.compile(
    r"(account\s*(?:no\.|number|#))|balance|opened|closed",
    re.I,
)


def _has_account_fields(block: list[str]) -> bool:
    for line in block[1:4]:
        if ACCOUNT_FIELD_RE.search(line):
            return True
    return False


def extract_account_blocks(text: str, debug: bool = False) -> list[list[str]]:
    """Return blocks of lines corresponding to individual accounts."""

    raw_lines = text.splitlines()
    lines: list[str] = []
    i = 0
    while i < len(raw_lines):
        current = raw_lines[i].strip()
        if i + 1 < len(raw_lines):
            nxt = raw_lines[i + 1].strip()
            if (
                current.isupper()
                and nxt.isupper()
                and len(current.split()) <= 2
                and len(nxt.split()) <= 2
            ):
                merged = f"{current} {nxt}"
                if debug:
                    print(
                        f"[~] Merged split heading '{current}' + '{nxt}' -> '{merged}'"
                    )
                lines.append(merged)
                i += 2
                continue
        lines.append(current)
        i += 1

    blocks: list[list[str]] = []
    current_block: list[str] = []
    capturing = False
    await_equifax_counts = False

    for idx, line in enumerate(lines):
        if _potential_account_name(line):
            if capturing:
                if _has_account_fields(current_block):
                    blocks.append(current_block)
                elif debug:
                    print(f"[~] Discarded block '{current_block[0]}' (no account fields)")
            current_block = [line]
            capturing = True
            await_equifax_counts = False
            if debug:
                print(f"[+] Start block '{line}'")
            continue

        if not capturing:
            continue

        current_block.append(line)

        if re.match(r"Equifax\b", line, re.I):
            await_equifax_counts = True
            if all(k in line for k in ("30:", "60:", "90:")):
                if _has_account_fields(current_block):
                    blocks.append(current_block)
                elif debug:
                    print(f"[~] Discarded block '{current_block[0]}' (no account fields)")
                if debug:
                    print(
                        f"[🏁] End block '{current_block[0]}' after Equifax counts line"
                    )
                current_block = []
                capturing = False
                await_equifax_counts = False
            continue

        if await_equifax_counts and all(k in line for k in ("30:", "60:", "90:")):
            if _has_account_fields(current_block):
                blocks.append(current_block)
            elif debug:
                print(f"[~] Discarded block '{current_block[0]}' (no account fields)")
            if debug:
                print(f"[🏁] End block '{current_block[0]}' after Equifax counts line")
            current_block = []
            capturing = False
            await_equifax_counts = False
            continue

    if capturing and current_block:
        if _has_account_fields(current_block):
            blocks.append(current_block)
        elif debug:
            print(f"[~] Discarded block '{current_block[0]}' (no account fields)")
        if debug:
            print(f"[🏁] End block '{current_block[0]}' (EOF)")

    return blocks


def parse_late_history_from_block(
    block: list[str], debug: bool = False
) -> dict[str, dict[str, int]]:
    """Parse bureau late-payment counts from a single account block."""

    details: dict[str, dict[str, int]] = {}
    pending_bureau: str | None = None
    found_bureau = False

    for line in block:
        clean = line.strip()
        bureau_match = re.match(r"(TransUnion|Experian|Equifax)\s*:?(.*)", clean, re.I)
        if bureau_match:
            bureau = bureau_match.group(1).title()
            rest = bureau_match.group(2)
            counts = _parse_late_counts(rest)
            found_bureau = True
            if counts:
                details[bureau] = counts
                pending_bureau = None
            else:
                pending_bureau = bureau
            continue

        if pending_bureau:
            counts = _parse_late_counts(clean)
            if counts:
                details[pending_bureau] = counts
            else:
                if debug:
                    print(
                        f"[~] Missing counts for {pending_bureau} in block starting '{block[0]}'"
                    )
            pending_bureau = None

    if not found_bureau and debug:
        print(f"[~] No bureau lines found in block starting '{block[0]}'")

    return details


def extract_late_history_blocks(
    text: str,
    known_accounts: set[str] | None = None,
    return_raw_map: bool = False,
    debug: bool = False,
    timeout: int = 4,
) -> dict:
    """Parse late payment history blocks and link them to accounts."""

    account_map: dict[str, dict[str, dict[str, int]]] = {}
    raw_map: dict[str, str] = {}

    def norm(name: str) -> str:
        from .generate_goodwill_letters import normalize_creditor_name

        return normalize_creditor_name(name)

    normalized_accounts = {norm(n): n for n in known_accounts or []}

    for block in extract_account_blocks(text, debug=debug):
        if not block:
            continue
        heading_raw = block[0].strip()
        acc_norm = norm(heading_raw)

        if known_accounts and acc_norm not in normalized_accounts:
            from difflib import get_close_matches

            match = get_close_matches(
                acc_norm, normalized_accounts.keys(), n=1, cutoff=0.8
            )
            if not match:
                if debug:
                    print(f"[~] Skipping unrecognized account '{heading_raw}'")
                continue
            if debug:
                print(f"[~] Fuzzy matched '{acc_norm}' → '{match[0]}'")
            acc_norm = match[0]

        details = parse_late_history_from_block(block, debug=debug)
        if not details:
            if debug:
                print(f"[~] Dropped candidate '{acc_norm}' (no details)")
            continue

        if not GENERIC_NAME_RE.search(acc_norm):
            account_map[acc_norm] = details
            raw_map.setdefault(acc_norm, heading_raw)
            if debug:
                found = sorted(details.keys())
                missing = [
                    b for b in {"Transunion", "Experian", "Equifax"} if b not in details
                ]
                print(
                    f"[🏁] End block '{heading_raw}' found={found or []} missing={missing or []}"
                )
                print(f"[📋] Parsed block '{heading_raw}' → {details}")

    for norm_name, bureaus in account_map.items():
        raw_name = raw_map.get(norm_name, norm_name)
        print(f"[📋] Parsed block '{raw_name}' → {bureaus}")

    if return_raw_map:
        return account_map, raw_map
    return account_map


def _total_lates(info) -> int:
    """Return the sum of all late payment counts across bureaus."""
    total = 0
    if isinstance(info, dict):
        for bureau_vals in info.values():
            if isinstance(bureau_vals, dict):
                for v in bureau_vals.values():
                    try:
                        total += int(v)
                    except (TypeError, ValueError):
                        continue
    return total


def has_late_indicator(acc: dict) -> bool:
    """Return True if account has explicit late payment info or matching text."""
    late = acc.get("late_payments")
    total_lates = _total_lates(late)
    if total_lates > 0:
        return True
    if "Late Payments" in acc.get("flags", []):
        # Some bureaus mark accounts with this flag even when no counts are
        # reported; ignore the flag if all counts are zero
        return False
    text = " ".join(
        str(acc.get(f, "")) for f in ["status", "remarks", "advisor_comment", "flags"]
    )
    if NO_LATE_PATTERN.search(text.lower()):
        return False
    return _has_late_flag(text)


CHARGEOFF_RE = re.compile(r"charge[- ]?off", re.I)
COLLECTION_RE = re.compile(r"collection", re.I)


def enforce_collection_status(acc: dict) -> None:
    """Ensure accounts mentioning both charge-off and collection are tagged as a collection.

    The original status string from the credit report is preserved in
    ``reported_status`` so downstream logic (e.g., letter generation) can display
    the exact wording. Only classification fields like ``account_type`` and
    ``flags`` are modified.
    """

    text = " ".join(
        str(acc.get(field, ""))
        for field in [
            "status",
            "remarks",
            "account_type",
            "account_status",
            "advisor_comment",
            "flags",
            "tags",
        ]
    ).lower()

    if CHARGEOFF_RE.search(text) and COLLECTION_RE.search(text):
        if acc.get("status") and "reported_status" not in acc:
            acc["reported_status"] = acc["status"]
        # Preserve the status field so the full text can be shown in letters.
        if acc.get("account_type") and "collection" not in str(acc["account_type"]).lower():
            acc["account_type"] = "Collection"
        else:
            acc.setdefault("account_type", "Collection")
        for field in ("flags", "tags"):
            val = acc.get(field)
            if val is None:
                acc[field] = ["Collection"]
            elif isinstance(val, list):
                if not any("collection" in str(v).lower() for v in val):
                    val.append("Collection")
            else:
                if "collection" not in str(val).lower():
                    acc[field] = [val, "Collection"]


INQUIRY_RE = re.compile(
    r"(?P<creditor>[A-Za-z0-9 .,'&/-]{3,})\s+(?P<date>\d{1,2}/\d{2,4})\s+(?P<bureau>TransUnion|Experian|Equifax)",
    re.I,
)

# Additional patterns for structured inquiry section parsing
INQ_HEADER_RE = re.compile(
    r"Creditor Name\s+Date of Inquiry\s+Credit Bureau",
    re.I,
)
INQ_LINE_RE = re.compile(
    r"(?P<creditor>[A-Za-z0-9 .,'&/-]+?)\s+(?P<date>\d{1,2}/\d{1,2}/\d{2,4})\s+(?P<bureau>(?:TransUnion|Trans Union|TU|Experian|EX|Equifax|EQ))",
    re.I,
)


def extract_inquiries(text: str) -> list[dict]:
    """Return inquiry tuples parsed from the text."""

    lines = [ln.strip() for ln in text.splitlines()]
    start = None
    for i, line in enumerate(lines):
        if line == "Inquiries":
            for j in (i + 1, i + 2):
                if j < len(lines) and INQ_HEADER_RE.search(lines[j]):
                    start = j + 1
                    break
            if start is not None:
                break

    found: list[dict] = []
    if start is not None:
        for line in lines[start:]:
            if line.startswith("Creditor Contacts"):
                break
            m = INQ_LINE_RE.search(line)
            if m:
                found.append(
                    {
                        "creditor_name": m.group("creditor").strip(),
                        "date": m.group("date"),
                        "bureau": normalize_bureau_name(m.group("bureau")),
                    }
                )

    if not found:
        compact = re.sub(r"\s+", " ", text)
        for m in INQUIRY_RE.finditer(compact):
            found.append(
                {
                    "creditor_name": m.group("creditor").strip(),
                    "date": m.group("date"),
                    "bureau": normalize_bureau_name(m.group("bureau")),
                }
            )

    return found


def filter_sections_by_bureau(sections, bureau_name, log_list=None):
    """Return relevant subsets only for the specified bureau.

    ``log_list`` if provided will be appended with human readable
    explanations when items are skipped or categorised.
    """
    bureau_name = normalize_bureau_name(bureau_name)

    filtered = {"disputes": [], "goodwill": [], "inquiries": [], "high_utilization": []}

    for acc in sections.get("negative_accounts", []):
        reported = [normalize_bureau_name(b) for b in acc.get("bureaus", [])]
        if bureau_name in reported:
            filtered["disputes"].append(acc)
        elif log_list is not None:
            log_list.append(
                f"[{bureau_name}] Skipped negative account '{acc.get('name')}' - not reported to this bureau"
            )

    for acc in sections.get("open_accounts_with_issues", []):
        reported = [normalize_bureau_name(b) for b in acc.get("bureaus", [])]
        if bureau_name in reported:
            if acc.get("goodwill_candidate", False):
                filtered["goodwill"].append(acc)
            else:
                filtered["disputes"].append(acc)
        elif log_list is not None:
            log_list.append(
                f"[{bureau_name}] Skipped account '{acc.get('name')}' - not reported to this bureau"
            )

    for acc in sections.get("high_utilization_accounts", []):
        reported = [normalize_bureau_name(b) for b in acc.get("bureaus", [])]
        if bureau_name in reported:
            filtered["high_utilization"].append(acc)
        elif log_list is not None:
            log_list.append(
                f"[{bureau_name}] Skipped high utilization account '{acc.get('name')}' - not reported to this bureau"
            )

    for inquiry in sections.get("inquiries", []):
        inquiry_bureau = normalize_bureau_name(inquiry.get("bureau"))
        if inquiry_bureau == bureau_name:
            filtered["inquiries"].append(inquiry)
        elif log_list is not None:
            if inquiry.get("bureau"):
                log_list.append(
                    f"[{bureau_name}] Skipped inquiry '{inquiry.get('creditor_name')}' - belongs to {inquiry.get('bureau')}"
                )

    # 🔍 Detect late payment indicators in positive or uncategorized accounts
    seen = {
        (
            normalize_creditor_name(acc.get("name", "")),
            acc.get("account_number"),
            bureau_name,
        )
        for section in filtered.values()
        for acc in section
        if isinstance(acc, dict)
    }

    extra_sources = sections.get("positive_accounts", []) + sections.get(
        "all_accounts", []
    )
    for acc in extra_sources:
        reported = [normalize_bureau_name(b) for b in acc.get("bureaus", [])]
        if bureau_name not in reported:
            continue
        key = (
            normalize_creditor_name(acc.get("name", "")),
            acc.get("account_number"),
            bureau_name,
        )
        if key in seen:
            continue
        if has_late_indicator(acc):
            enriched = acc.copy()
            text = " ".join(
                str(acc.get(field, ""))
                for field in ["status", "remarks", "advisor_comment", "flags"]
            )
            if (
                "good standing" in text.lower()
                or "closed" in str(acc.get("account_status", "")).lower()
            ):
                enriched["goodwill_candidate"] = True
                filtered["goodwill"].append(enriched)
            else:
                filtered["disputes"].append(enriched)
            seen.add(key)

    return filtered


def convert_txts_to_pdfs(folder: Path):
    """
    Converts .txt files in the given folder to styled PDFs with Unicode support.
    """
    txt_files = list(folder.glob("*.txt"))
    output_folder = folder / "converted"
    output_folder.mkdir(exist_ok=True)

    for txt_path in txt_files:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        font_path = "fonts/DejaVuSans.ttf"
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.add_font("DejaVu", "B", font_path, uni=True)
        pdf.set_font("DejaVu", "B", 14)

        title = txt_path.stem
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("DejaVu", "", 12)

        with open(txt_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    pdf.ln(5)
                    continue
                try:
                    pdf.multi_cell(0, 10, line)
                except Exception as e:
                    print(f"[⚠️] Failed to render line: {line[:50]} — {e}")
                    continue

        new_path = output_folder / (txt_path.stem + ".pdf")
        pdf.output(str(new_path))
        print(f"[📄] Converted to PDF: {new_path}")


def extract_pdf_text_safe(pdf_path: Path, max_chars: int = 4000) -> str:
    """Extract text from a PDF using pdfplumber with a fitz fallback."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            parts = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    parts.append(text)
                if sum(len(p) for p in parts) >= max_chars:
                    break
            joined = "\n".join(parts)
            if joined:
                return joined[:max_chars]
    except Exception as e:
        print(f"[⚠️] pdfplumber failed for {pdf_path}: {e}")

    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) >= max_chars:
                break
        doc.close()
        return text[:max_chars]
    except Exception as e:
        print(f"[❌] Fallback extraction failed for {pdf_path}: {e}")
        return ""


def gather_supporting_docs(
    session_id: str, max_chars: int = 4000
) -> tuple[str, list[str], dict[str, str]]:
    """Return a summary text, list of filenames and mapping of snippets for supplemental PDFs.

    Each PDF found under ``supporting_docs/{session_id}`` (and the base
    ``supporting_docs`` folder as a fallback) is parsed individually. A short
    snippet of its text is included in the summary. The combined summary is
    truncated if it exceeds ``max_chars`` (roughly ~1000 tokens).
    """

    base = Path("supporting_docs")
    candidates = []
    if session_id:
        candidates.append(base / session_id)
    candidates.append(base)

    summaries = []
    filenames = []
    doc_snippets: dict[str, str] = {}
    total_len = 0

    for folder in candidates:
        if not folder.exists():
            continue
        for pdf_path in sorted(folder.glob("*.pdf")):
            if total_len >= max_chars:
                print(f"[⚠️] Reached max characters, truncating remaining docs.")
                break
            try:
                raw_text = extract_pdf_text_safe(pdf_path, 1500)
                snippet = " ".join(raw_text.split())[:700] if raw_text else ""
                if snippet:
                    summary = (
                        f"The following document was provided: '{pdf_path.name}'\n"
                        f"→ Summary: {snippet}"
                    )
                    summaries.append(summary)
                    doc_snippets[pdf_path.name] = snippet
                    total_len += len(summary) + 1
                filenames.append(pdf_path.name)
                print(f"[📎] Parsed supporting doc: {pdf_path.name}")
            except Exception as e:
                print(f"[⚠️] Failed to parse {pdf_path.name}: {e}")
                continue
        if total_len >= max_chars:
            break

    combined = "\n".join(summaries)
    if len(combined) > max_chars:
        combined = combined[:max_chars]
        print("[⚠️] Combined supporting docs summary truncated due to length.")

    return combined.strip(), filenames, doc_snippets


def gather_supporting_docs_text(session_id: str, max_chars: int = 4000) -> str:
    """Backward compatible wrapper returning only the summary text."""
    summary, _, _ = gather_supporting_docs(session_id, max_chars)
    return summary


def extract_summary_from_sections(sections):
    """
    Returns analytical summary from full data structure.
    """
    summary = {
        "total_negative": len(sections.get("negative_accounts", [])),
        "total_late_payments": len(
            [
                acc
                for acc in sections.get("open_accounts_with_issues", [])
                if has_late_indicator(acc)
            ]
        ),
        "high_utilization_accounts": len(sections.get("high_utilization_accounts", [])),
        "recent_inquiries": len(sections.get("inquiries", [])),
        "identity_theft_suspicions": len(
            [
                acc
                for acc in sections.get("negative_accounts", [])
                if acc.get("is_suspected_identity_theft")
            ]
        ),
        "by_bureau": {
            bureau: {
                "disputes": len(
                    [
                        acc
                        for acc in sections.get("negative_accounts", [])
                        if bureau in acc.get("bureaus", [])
                    ]
                ),
                "goodwill": len(
                    [
                        acc
                        for acc in sections.get("open_accounts_with_issues", [])
                        if acc.get("goodwill_candidate")
                        and bureau in acc.get("bureaus", [])
                    ]
                ),
                "high_utilization": len(
                    [
                        acc
                        for acc in sections.get("high_utilization_accounts", [])
                        if bureau in acc.get("bureaus", [])
                    ]
                ),
            }
            for bureau in BUREAUS
        },
    }
    return summary
