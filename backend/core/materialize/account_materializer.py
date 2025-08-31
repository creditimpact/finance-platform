"""Account materializer (assemble-only).

Builds full account JSONs for a given session by selecting existing fields
from OCR/Parser outputs, and attaches analyzer tags in a derived block.

Scope and constraints:
- Do NOT derive/compute new values (no parsing, no inference, no number
  extraction, no status recalculation). Merely collect fields that already
  exist in the provided OCR document structure.
- Only include accounts present in analyzer_problems (by account_id when
  available, otherwise by normalized/name match case/space-insensitive).
- Attach analyzer tags under derived.analyzer_tags and (optionally) mirror
  primary_issue at top level if the source lacks it or it is unknown.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping
import logging
import re
from typing import Optional
from datetime import datetime

from backend.core.logic.report_analysis.report_parsing import (
    _empty_bureau_map,
    _find_block_lines_for_account,
    parse_account_block,
    parse_collection_block,
    _fill_bureau_map_from_sources,
    ACCOUNT_FIELD_SET,
)
from backend.core.logic.report_analysis.normalize import to_number, to_iso_date

logger = logging.getLogger(__name__)


def _norm_name(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(str(s).strip().lower().split())


def _slug(s: str | None) -> str:
    s = (s or "").strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-]", "", s) or "account"


# ---------------------------------------------------------------------------
# Bureau-complete view (assemble-only)
# ---------------------------------------------------------------------------

BUREAUS = ("transunion", "experian", "equifax")
FIELDS = (
    "payment_status",
    "account_status",
    "remarks",
    "creditor_remarks",
    "past_due_amount",
    "balance",
    "credit_limit",
    "last_reported",
    "date_opened",
    "date_reported",
    "account_number_display",
    "account_number_last4",
)

# ACCOUNT_FIELD_SET imported from report_parsing


def _norm_bureau_key(s: str | None) -> str:
    key = (s or "").strip().lower()
    if key in ("tu", "transunion"):  # accept short code
        return "transunion"
    if key in ("ex", "experian"):
        return "experian"
    if key in ("eq", "equifax"):
        return "equifax"
    return key


def _to_number(val: Any) -> Optional[float]:  # gentle normalization
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    # remove currency symbols/commas/spaces
    cleaned = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(cleaned) if cleaned else None
    except Exception:
        return None


def _to_iso_date(val: Any) -> Optional[str]:  # gentle normalization
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # Already ISO YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    # MM/DD/YYYY or M/D/YYYY → YYYY-MM-DD
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        mm = int(m.group(1))
        dd = int(m.group(2))
        yyyy = int(m.group(3))
        return f"{yyyy:04d}-{mm:02d}-{dd:02d}"
    # Fallback: return original string
    return s


def _build_bureau_complete(acc: dict) -> dict[str, dict[str, Any]]:
    by_field: dict[str, dict[str, Any]] = {b: {f: None for f in FIELDS} for b in BUREAUS}

    # Map bureaus[] entries by normalized name
    by_name: dict[str, dict] = {}
    for entry in acc.get("bureaus") or []:
        if not isinstance(entry, dict):
            continue
        bk = _norm_bureau_key(entry.get("bureau") or entry.get("name"))
        if bk in BUREAUS:
            by_name[bk] = entry

    # Flat maps
    ps_map = {
        _norm_bureau_key(k): v for k, v in (acc.get("payment_statuses") or {}).items()
    }
    bs_map = {
        _norm_bureau_key(k): v for k, v in (acc.get("bureau_statuses") or {}).items()
    }
    details_map_raw = acc.get("bureau_details") or {}
    details_map = { _norm_bureau_key(k): v for k, v in details_map_raw.items() if isinstance(v, dict) }

    # Common top-level fallbacks
    acc_num_disp = acc.get("account_number_display")
    acc_num_last4 = acc.get("account_number_last4")

    for b in BUREAUS:
        out = by_field[b]
        src = by_name.get(b) or {}
        det = details_map.get(b) or {}

        # 1) Copy from bureaus[] entry when available (as-is, but normalize numeric/date for view)
        if isinstance(src, dict):
            if src.get("payment_status") not in (None, "", {}, []):
                out["payment_status"] = src.get("payment_status")
            if src.get("account_status") not in (None, "", {}, []):
                out["account_status"] = src.get("account_status")
            if src.get("remarks") not in (None, "", {}, []):
                out["remarks"] = src.get("remarks")
            if src.get("creditor_remarks") not in (None, "", {}, []):
                out["creditor_remarks"] = src.get("creditor_remarks")
            if src.get("past_due_amount") not in (None, "", {}, []):
                out["past_due_amount"] = _to_number(src.get("past_due_amount"))
            if src.get("balance") not in (None, "", {}, []):
                out["balance"] = _to_number(src.get("balance"))
            if src.get("credit_limit") not in (None, "", {}, []):
                out["credit_limit"] = _to_number(src.get("credit_limit"))
            if src.get("last_reported") not in (None, "", {}, []):
                out["last_reported"] = _to_iso_date(src.get("last_reported"))
            if src.get("date_opened") not in (None, "", {}, []):
                out["date_opened"] = _to_iso_date(src.get("date_opened"))
            if src.get("date_reported") not in (None, "", {}, []):
                out["date_reported"] = _to_iso_date(src.get("date_reported"))

        # 2) Flat maps for payment/account status
        if out["payment_status"] is None and b in ps_map and ps_map[b] not in (None, "", {}):
            out["payment_status"] = ps_map[b]
        if out["account_status"] is None and b in bs_map and bs_map[b] not in (None, "", {}):
            out["account_status"] = bs_map[b]

        # 3) Bureau details for financials/dates
        if isinstance(det, dict):
            if out["past_due_amount"] is None and det.get("past_due_amount") not in (None, "", {}):
                out["past_due_amount"] = _to_number(det.get("past_due_amount"))
            if out["balance"] is None and det.get("balance") not in (None, "", {}):
                out["balance"] = _to_number(det.get("balance"))
            if out["credit_limit"] is None and det.get("credit_limit") not in (None, "", {}):
                out["credit_limit"] = _to_number(det.get("credit_limit"))
            if out["last_reported"] is None and det.get("last_reported") not in (None, "", {}):
                out["last_reported"] = _to_iso_date(det.get("last_reported"))
            if out["date_opened"] is None and det.get("date_opened") not in (None, "", {}):
                out["date_opened"] = _to_iso_date(det.get("date_opened"))
            if out["date_reported"] is None and det.get("date_reported") not in (None, "", {}):
                out["date_reported"] = _to_iso_date(det.get("date_reported"))

        # 4) Cross-bureau consistent fields (safe replication): account number
        out["account_number_display"] = out["account_number_display"] or acc_num_disp
        out["account_number_last4"] = out["account_number_last4"] or acc_num_last4

    # If raw.account_history.by_bureau exists, use its balance_owed as overlay 'balance' when missing
    try:
        raw_by_bureau = (
            acc.get("raw", {})
            .get("account_history", {})
            .get("by_bureau", {})
        )
        for b in BUREAUS:
            if by_field.get(b, {}).get("balance") is None:
                val = raw_by_bureau.get(b, {}).get("balance_owed") if isinstance(raw_by_bureau, dict) else None
                if val not in (None, "", {}, []):
                    by_field[b]["balance"] = _to_number(val)
    except Exception:
        pass

    return by_field


def _build_account_history_by_bureau(acc: dict) -> dict[str, dict[str, Any]]:
    """Assemble raw.account_history.by_bureau filled with the full field set.

    Values are taken from existing structures only (no parsing/derivations):
    - acc['bureaus'] entries per bureau
    - acc['bureau_details'][bureau]
    - flat maps when applicable
    Missing values remain None.
    Gentle normalization for numbers/dates within this view only.
    """

    by_hist: dict[str, dict[str, Any]] = {b: {f: None for f in ACCOUNT_FIELD_SET} for b in BUREAUS}

    # Map bureaus[] entries by normalized name
    by_name: dict[str, dict] = {}
    for entry in acc.get("bureaus") or []:
        if isinstance(entry, dict):
            bk = _norm_bureau_key(entry.get("bureau") or entry.get("name"))
            if bk in BUREAUS:
                by_name[bk] = entry

    details_map_raw = acc.get("bureau_details") or {}
    details_map = { _norm_bureau_key(k): v for k, v in details_map_raw.items() if isinstance(v, dict) }

    acc_num_disp = acc.get("account_number_display")
    acc_num_last4 = acc.get("account_number_last4")

    for b in BUREAUS:
        out = by_hist[b]
        src = by_name.get(b) or {}
        det = details_map.get(b) or {}

        # Prefer bureaus[] source first
        def _copy_src(key: str, norm: Optional[str] = None, numeric: bool = False, date: bool = False):
            val = src.get(key)
            if val not in (None, "", {}, []):
                if numeric:
                    return _to_number(val)
                if date:
                    return _to_iso_date(val)
                return val
            # try from details
            val2 = det.get(norm or key)
            if val2 not in (None, "", {}, []):
                if numeric:
                    return _to_number(val2)
                if date:
                    return _to_iso_date(val2)
                return val2
            return None

        out["account_number_display"] = src.get("account_number_display") or acc_num_disp
        out["account_number_last4"] = src.get("account_number_last4") or acc_num_last4
        out["high_balance"] = _copy_src("high_balance", numeric=True)
        out["last_verified"] = _copy_src("last_verified", date=True)
        out["date_of_last_activity"] = _copy_src("date_of_last_activity", date=True)
        out["date_reported"] = _copy_src("date_reported", date=True)
        out["date_opened"] = _copy_src("date_opened", date=True)
        # Balance owed may be named 'balance' in details or src
        out["balance_owed"] = _copy_src("balance_owed", norm="balance", numeric=True) or _copy_src("balance", numeric=True)
        out["closed_date"] = _copy_src("closed_date", date=True)
        out["account_rating"] = _copy_src("account_rating")
        out["account_description"] = _copy_src("account_description")
        out["dispute_status"] = _copy_src("dispute_status")
        out["creditor_type"] = _copy_src("creditor_type")
        out["account_status"] = _copy_src("account_status")
        out["payment_status"] = _copy_src("payment_status")
        out["creditor_remarks"] = _copy_src("creditor_remarks")
        out["payment_amount"] = _copy_src("payment_amount", numeric=True)
        out["last_payment"] = _copy_src("last_payment", date=True)
        out["term_length"] = _copy_src("term_length")
        out["past_due_amount"] = _copy_src("past_due_amount", numeric=True)
        out["account_type"] = _copy_src("account_type")
        out["payment_frequency"] = _copy_src("payment_frequency")
        out["credit_limit"] = _copy_src("credit_limit", numeric=True)
        out["two_year_payment_history"] = src.get("two_year_payment_history") or det.get("two_year_payment_history")
        out["seven_year_days_late"] = src.get("seven_year_days_late") or det.get("seven_year_days_late")

    return by_hist


def _to_list(obj: Any) -> list:
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    return [obj]


def _collect_sources_map(ocr_doc: Any) -> dict[str, list[dict]]:
    """Return source containers for assembling accounts, in strict order.

    Order: all_accounts → accounts → problem_accounts (last resort).
    The function does not derive; it only exposes existing containers.
    """

    out: dict[str, list[dict]] = {"all_accounts": [], "accounts": [], "problem_accounts": []}

    if isinstance(ocr_doc, list):
        # Non-structured list of dicts treated as 'accounts'
        out["accounts"] = [x for x in ocr_doc if isinstance(x, dict)]
        return out

    if not isinstance(ocr_doc, Mapping):
        return out

    for key in ("all_accounts", "accounts", "problem_accounts"):
        items = ocr_doc.get(key)
        if isinstance(items, list):
            out[key] = [x for x in items if isinstance(x, dict)]

    return out


def _build_index(accounts: Iterable[Mapping[str, Any]]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Return indices (by id, by normalized name) for quick matching."""

    by_id: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    for a in accounts:
        if not isinstance(a, Mapping):
            continue
        acc = dict(a)
        acc_id = str(acc.get("account_id") or "").strip()
        if acc_id:
            by_id[acc_id] = acc
        nm = _norm_name(acc.get("normalized_name") or acc.get("name"))
        if nm:
            by_name[nm] = acc
    return by_id, by_name


def _mk_minimal_from_analyzer(p: Mapping[str, Any]) -> dict:
    """Fallback minimal structure when no source account is found in OCR doc."""
    out = {
        "account_id": p.get("account_id"),
        "name": p.get("name") or p.get("normalized_name"),
        "normalized_name": p.get("normalized_name") or _norm_name(p.get("name")),
    }
    # Attach analyzer tags only
    tags = {
        "primary_issue": p.get("primary_issue"),
        "issue_types": p.get("issue_types"),
        "problem_reasons": p.get("problem_reasons"),
        "flags": p.get("flags"),
        "advisor_comment": p.get("advisor_comment"),
    }
    out["derived"] = {"analyzer_tags": tags}
    # Optional top-level mirror without overriding source fields
    if not out.get("primary_issue") and tags.get("primary_issue"):
        out["primary_issue"] = tags.get("primary_issue")
    return out


def materialize_accounts(
    session_id: str,
    ocr_doc: dict | list[dict] | str,
    analyzer_problems: list[dict],
) -> list[dict]:
    """Build full account JSONs from existing OCR/Parser data.

    Rules:
    - Do NOT parse or derive new fields. Only pick fields that already exist
      in the provided OCR/Parser structures.
    - Only include accounts listed in `analyzer_problems`.
    - Attach analyzer tags under `derived.analyzer_tags`.
    - Optionally mirror `primary_issue` at top-level if missing/unknown.
    """

    # Collect and index source accounts from the OCR document
    sources_map = _collect_sources_map(ocr_doc)
    by_id_all, by_name_all = _build_index(sources_map.get("all_accounts", []))
    by_id_acc, by_name_acc = _build_index(sources_map.get("accounts", []))
    by_id_prob, by_name_prob = _build_index(sources_map.get("problem_accounts", []))

    logger.info(
        "materializer_sources session=%s containers=%s problems=%d",
        session_id,
        ",".join(k for k, v in sources_map.items() if v),
        len(analyzer_problems or []),
    )

    out: list[dict] = []

    matched_from = {"all_accounts": 0, "accounts": 0, "problem_accounts": 0, "minimal": 0}

    for p in analyzer_problems or []:
        if not isinstance(p, Mapping):
            continue
        pid = str(p.get("account_id") or "").strip()
        pname = _norm_name(p.get("normalized_name") or p.get("name"))

        src: dict | None = None
        matched_key = None

        # all_accounts by id then name
        if pid and pid in by_id_all:
            src, matched_key = dict(by_id_all[pid]), "all_accounts"
        elif pname and pname in by_name_all:
            src, matched_key = dict(by_name_all[pname]), "all_accounts"
        # accounts by id then name
        elif pid and pid in by_id_acc:
            src, matched_key = dict(by_id_acc[pid]), "accounts"
        elif pname and pname in by_name_acc:
            src, matched_key = dict(by_name_acc[pname]), "accounts"
        # problem_accounts (last resort) by id then name
        elif pid and pid in by_id_prob:
            src, matched_key = dict(by_id_prob[pid]), "problem_accounts"
        elif pname and pname in by_name_prob:
            src, matched_key = dict(by_name_prob[pname]), "problem_accounts"

        if src is None:
            # No source found: produce minimal doc from analyzer info only
            md = _mk_minimal_from_analyzer(p)
            out.append(md)
            matched_from["minimal"] += 1
            logger.info(
                "materialize_match session=%s account_id=%s name=%s source=%s",
                session_id,
                pid or "",
                (p.get("normalized_name") or p.get("name") or ""),
                "minimal",
            )
            continue

        # Attach analyzer tags without modifying existing source fields
        tags = {
            "primary_issue": p.get("primary_issue"),
            "issue_types": p.get("issue_types"),
            "problem_reasons": p.get("problem_reasons"),
            "flags": p.get("flags"),
            "advisor_comment": p.get("advisor_comment"),
        }
        derived = dict(src.get("derived") or {})
        derived["analyzer_tags"] = tags
        src["derived"] = derived

        # Ensure IDs/names are present and aligned
        # Proper fix: set account_id to slug(normalized_name|name)
        slug_id = _slug(p.get("normalized_name") or p.get("name") or src.get("normalized_name") or src.get("name"))
        if slug_id:
            src["account_id"] = slug_id
        if not src.get("normalized_name") and pname:
            src["normalized_name"] = pname
        if not src.get("name") and p.get("name"):
            src["name"] = p.get("name")

        # Preserve fingerprint if present in source; analyzer may also include
        if not src.get("account_fingerprint") and p.get("account_fingerprint"):
            src["account_fingerprint"] = p.get("account_fingerprint")

        # Optional top-level mirror for compatibility (no overwrite)
        if (src.get("primary_issue") in (None, "", "unknown")) and tags.get("primary_issue"):
            src["primary_issue"] = tags.get("primary_issue")

        # Build raw scaffold and account_history.by_bureau
        try:
            raw = dict(src.get("raw") or {})
            # Personal information (placeholders unless parser provided)
            raw.setdefault("personal_information", {
                "name": None,
                "aka": None,
                "dob": None,
                "current_address": None,
                "previous_addresses": None,
                "employer": None,
                "_provenance": {},
            })
            # Summary (report-level metrics)
            raw.setdefault("summary", {"_provenance": {}})
            # Account history by bureau (full field set)
            raw.setdefault("account_history", {})

            # Initialize empty 25-field maps for each bureau and attempt to fill
            # them using available sources and OCR block lines.
            by = {b: _empty_bureau_map() for b in BUREAUS}

            # --- BEGIN: unified SID ---
            sections = locals().get("sections")
            sid = (
                (ocr_doc.get("session_id") if isinstance(ocr_doc, Mapping) else None)
                or (ocr_doc.get("request_id") if isinstance(ocr_doc, Mapping) else None)
                or (sections.get("session_id") if isinstance(sections, Mapping) else None)
                or (sections.get("request_id") if isinstance(sections, Mapping) else None)
            )
            # --- END: unified SID ---

            # --- BEGIN: MAT fallback diagnostics ---
            try:
                from pathlib import Path
                import json

                blkdir = Path("traces") / "blocks" / (sid or "")
                dir_exists = blkdir.is_dir()
                files_count = len(list(blkdir.glob("block_*.json"))) if dir_exists else 0
                logger.warning(
                    "MAT: fallback sid=%s dir=%s exists=%s files=%d",
                    sid,
                    str(blkdir),
                    dir_exists,
                    files_count,
                )

                if not ocr_doc.get("fbk_blocks") and dir_exists and files_count > 0:
                    blocks = []
                    for p in sorted(blkdir.glob("block_*.json")):
                        with p.open("r", encoding="utf-8") as f:
                            blocks.append(json.load(f))
                    if blocks:
                        ocr_doc["fbk_blocks"] = blocks
                        from backend.core.logic.report_analysis.analyze_report import build_block_fuzzy
                        ocr_doc["blocks_by_account_fuzzy"] = build_block_fuzzy(blocks)
                        logger.warning(
                            "MAT: rebuilt lines_map=%d from traces",
                            len((ocr_doc.get("blocks_by_account_fuzzy") or {}).keys()),
                        )
            except Exception:
                logger.exception("materializer_block_fallback_failed")

            logger.warning(
                "MAT: before-find lines_map=%d sid=%s",
                len((ocr_doc.get("blocks_by_account_fuzzy") or {}).keys()),
                sid,
            )
            # --- END: MAT fallback diagnostics ---

            lines = _find_block_lines_for_account(ocr_doc, src)
            for b in BUREAUS:
                try:
                    _fill_bureau_map_from_sources(src, b, by[b], lines)
                except Exception:
                    logger.exception("_fill_bureau_map_from_sources_failed")

            # --- BEGIN: parser-driven merge ---
            try:
                # 1) General parser for account table
                maps_acc = parse_account_block(lines or [])
            except Exception:
                logger.exception("parse_account_block_failed")
                maps_acc = {}

            try:
                # 2) Parser for collection/chargeoff (fills gaps)
                maps_col = parse_collection_block(lines or [])
            except Exception:
                logger.exception("parse_collection_block_failed")
                maps_col = {}

            for b in BUREAUS:
                # start with what we already have in by[b], then fill only None
                bm = maps_acc.get(b, {}) if isinstance(maps_acc, dict) else {}
                bm2 = maps_col.get(b, {}) if isinstance(maps_col, dict) else {}
                for key in ACCOUNT_FIELD_SET:
                    if by[b].get(key) is None and bm.get(key) is not None:
                        by[b][key] = bm[key]
                for key in ACCOUNT_FIELD_SET:
                    if by[b].get(key) is None and bm2.get(key) is not None:
                        by[b][key] = bm2[key]

            filled = {b: sum(1 for k in ACCOUNT_FIELD_SET if by[b].get(k) is not None) for b in BUREAUS}
            for b in BUREAUS:
                logger.warning(
                    "parser_bureau_fill account=%s bureau=%s filled=%d/25",
                    src.get("normalized_name") or src.get("name"),
                    b,
                    filled[b],
                )
            # --- END: parser-driven merge ---

            # Merge into raw.account_history.by_bureau without overriding
            # existing non-null values.
            ah = raw.setdefault("account_history", {})
            byb = ah.setdefault("by_bureau", {})
            for b in BUREAUS:
                dest = byb.setdefault(b, _empty_bureau_map())
                for k in ACCOUNT_FIELD_SET:
                    v = by[b].get(k)
                    if dest.get(k) is None and v is not None:
                        dest[k] = v

            # Public information / Inquiries arrays
            raw.setdefault("public_information", {"items": []})
            raw.setdefault("inquiries", {"items": []})
            # Populate raw.inquiries from OCR doc if present
            try:
                inqs = ocr_doc.get("inquiries") if isinstance(ocr_doc, Mapping) else []
                items: list[dict] = []
                for inq in inqs or []:
                    if not isinstance(inq, Mapping):
                        continue
                    bureau = _norm_bureau_key(inq.get("bureau")) if inq.get("bureau") else None
                    date = _to_iso_date(inq.get("date")) if inq.get("date") else None
                    items.append(
                        {
                            "bureau": bureau,
                            "subscriber": inq.get("creditor_name") or inq.get("subscriber") or inq.get("name"),
                            "date": date,
                            "type": inq.get("type"),
                            "permissible_purpose": inq.get("permissible_purpose"),
                            "remarks": inq.get("remarks"),
                            "_provenance": inq.get("_provenance", {}),
                        }
                    )
                if items:
                    raw["inquiries"]["items"] = items
                elif inqs:
                    logger.warning(
                        "inquiries_detected_but_not_written session=%s account=%s",
                        session_id,
                        src.get("account_id") or _slug(src.get("name")),
                    )
            except Exception:
                pass
            # Populate raw.public_information from OCR doc if present
            try:
                pub = ocr_doc.get("public_information") if isinstance(ocr_doc, Mapping) else []
                pitems: list[dict] = []
                for it in pub or []:
                    if not isinstance(it, Mapping):
                        continue
                    bureau = _norm_bureau_key(it.get("bureau")) if it.get("bureau") else None
                    date_filed = _to_iso_date(it.get("date_filed")) if it.get("date_filed") else None
                    amount = _to_number(it.get("amount")) if it.get("amount") else None
                    pitems.append(
                        {
                            "bureau": bureau,
                            "item_type": it.get("item_type") or it.get("type"),
                            "status": it.get("status"),
                            "date_filed": date_filed,
                            "amount": amount,
                            "remarks": it.get("remarks"),
                            "_provenance": it.get("_provenance", {}),
                        }
                    )
                if pitems:
                    raw["public_information"]["items"] = pitems
                elif pub:
                    logger.warning(
                        "public_info_detected_but_not_written session=%s account=%s",
                        session_id,
                        src.get("account_id") or _slug(src.get("name")),
                    )
            except Exception:
                pass
            src["raw"] = raw
            # --- ensure meta coverage log after merge ---
            by_bureau = raw.get("account_history", {}).get("by_bureau", {})
            tu = sum(
                1
                for f in ACCOUNT_FIELD_SET
                if by_bureau.get("transunion", {}).get(f) is not None
            )
            ex = sum(
                1
                for f in ACCOUNT_FIELD_SET
                if by_bureau.get("experian", {}).get(f) is not None
            )
            eq = sum(
                1
                for f in ACCOUNT_FIELD_SET
                if by_bureau.get("equifax", {}).get(f) is not None
            )
            logger.warning(
                "bureau_meta_coverage name=%s tu_filled=%d ex_filled=%d eq_filled=%d",
                src.get("normalized_name") or src.get("name"),
                tu,
                ex,
                eq,
            )
        except Exception:
            pass

        # Add bureau-complete view for UI (assemble-only overlay)
        try:
            bureaus_by_field = _build_bureau_complete(src)
            src["bureaus_by_field"] = bureaus_by_field
            missing_counts = {
                b: sum(1 for f in FIELDS if bureaus_by_field.get(b, {}).get(f) is None)
                for b in BUREAUS
            }
            logger.info(
                "materializer_bureau_coverage session=%s account_id=%s missing_counts=%s",
                session_id,
                src.get("account_id") or "",
                missing_counts,
            )
        except Exception:
            # Non-fatal: keep assembled src without the overlay
            pass

        # Stages.materializer coverage and timestamps
        try:
            stages = dict(src.get("stages") or {})
            mat = dict(stages.get("materializer") or {})
            mat["producer"] = mat.get("producer") or "materializer"
            ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            mat["ts"] = ts
            # Coverage stats: per-bureau missing and top missing fields overall
            by_bureau = src.get("raw", {}).get("account_history", {}).get("by_bureau", {})
            counts = {b: sum(1 for f in ACCOUNT_FIELD_SET if by_bureau.get(b, {}).get(f) is None) for b in BUREAUS}
            # Top missing fields across bureaus
            field_miss: dict[str, int] = {f: 0 for f in ACCOUNT_FIELD_SET}
            for b in BUREAUS:
                for f in ACCOUNT_FIELD_SET:
                    if by_bureau.get(b, {}).get(f) is None:
                        field_miss[f] += 1
            top_missing = sorted(field_miss.items(), key=lambda x: x[1], reverse=True)[:5]
            mat["coverage"] = {
                "transunion_missing": counts.get("transunion", 0),
                "experian_missing": counts.get("experian", 0),
                "equifax_missing": counts.get("equifax", 0),
                "top_missing_fields": [k for k, _ in top_missing],
            }
            stages["materializer"] = mat
            src["stages"] = stages
            # Set created_at/updated_at if absent
            if not src.get("created_at"):
                src["created_at"] = ts
            src["updated_at"] = ts
        except Exception:
            pass

        out.append(src)
        if matched_key:
            matched_from[matched_key] += 1
        logger.info(
            "materialize_match session=%s account_id=%s name=%s source=%s",
            session_id,
            pid or "",
            (p.get("normalized_name") or p.get("name") or ""),
            matched_key or "minimal",
        )

    logger.info(
        "materializer_stats session=%s matched_all=%d matched_accounts=%d matched_problem=%d minimal=%d",
        session_id,
        matched_from["all_accounts"],
        matched_from["accounts"],
        matched_from["problem_accounts"],
        matched_from["minimal"],
    )
    logger.info("materializer_output_count session=%s count=%d", session_id, len(out))
    return out
