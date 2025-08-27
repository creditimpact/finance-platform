"""Deterministic extractors for SmartCredit reports.

This package also re-exports a handful of legacy extractor helpers used by
existing code paths (e.g. :mod:`report_prompting`).
"""

from . import sections, tokens, accounts, report_meta, summary, legacy

# Legacy helpers ---------------------------------------------------------------
extract_account_number_masks = legacy.extract_account_number_masks
extract_account_statuses = legacy.extract_account_statuses
extract_dofd = legacy.extract_dofd
extract_inquiry_dates = legacy.extract_inquiry_dates

__all__ = [
    "sections",
    "tokens",
    "accounts",
    "report_meta",
    "summary",
    "extract_account_number_masks",
    "extract_account_statuses",
    "extract_dofd",
    "extract_inquiry_dates",
]
