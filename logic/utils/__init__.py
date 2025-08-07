"""Utility helpers split into focused submodules.

This package re-exports stable public helpers so legacy imports like
``from logic.utils import X`` continue to work.
"""
from __future__ import annotations

from .names_normalization import (
    BUREAUS,
    BUREAU_ALIASES,
    normalize_creditor_name,
    normalize_bureau_name,
)
from .file_paths import safe_filename
from .note_handling import (
    HARDSHIP_RE,
    is_general_hardship_note,
    analyze_custom_notes,
    get_client_address_lines,
)
from .text_parsing import (
    LATE_PATTERN,
    NO_LATE_PATTERN,
    GENERIC_NAME_RE,
    extract_account_blocks,
    parse_late_history_from_block,
    extract_late_history_blocks,
    has_late_indicator,
    CHARGEOFF_RE,
    COLLECTION_RE,
    enforce_collection_status,
)
from .inquiries import (
    INQUIRY_RE,
    INQ_HEADER_RE,
    INQ_LINE_RE,
    extract_inquiries,
)
from .pdf_ops import (
    convert_txts_to_pdfs,
    extract_pdf_text_safe,
    gather_supporting_docs,
    gather_supporting_docs_text,
)
from .report_sections import (
    filter_sections_by_bureau,
    extract_summary_from_sections,
)

from . import (
    names_normalization,
    note_handling,
    file_paths,
    text_parsing,
    inquiries,
    pdf_ops,
    report_sections,
)

__all__ = [
    "BUREAUS",
    "BUREAU_ALIASES",
    "normalize_creditor_name",
    "normalize_bureau_name",
    "safe_filename",
    "HARDSHIP_RE",
    "is_general_hardship_note",
    "analyze_custom_notes",
    "get_client_address_lines",
    "LATE_PATTERN",
    "NO_LATE_PATTERN",
    "GENERIC_NAME_RE",
    "extract_account_blocks",
    "parse_late_history_from_block",
    "extract_late_history_blocks",
    "has_late_indicator",
    "CHARGEOFF_RE",
    "COLLECTION_RE",
    "enforce_collection_status",
    "INQUIRY_RE",
    "INQ_HEADER_RE",
    "INQ_LINE_RE",
    "extract_inquiries",
    "convert_txts_to_pdfs",
    "extract_pdf_text_safe",
    "gather_supporting_docs",
    "gather_supporting_docs_text",
    "filter_sections_by_bureau",
    "extract_summary_from_sections",
    "names_normalization",
    "note_handling",
    "file_paths",
    "text_parsing",
    "inquiries",
    "pdf_ops",
    "report_sections",
]
