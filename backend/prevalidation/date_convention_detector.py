"""Utilities for detecting the month language in a run's Two-Year Payment History."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable


HEBREW_MONTHS = (
    "ינו׳",
    "פבר׳",
    "מרץ",
    "אפר׳",
    "מאי",
    "יוני",
    "יולי",
    "אוג׳",
    "ספט׳",
    "אוק׳",
    "נוב׳",
    "דצמ׳",
)

ENGLISH_MONTH_TOKENS = frozenset(
    (
        "jan",
        "january",
        "feb",
        "february",
        "mar",
        "march",
        "apr",
        "april",
        "may",
        "jun",
        "june",
        "jul",
        "july",
        "aug",
        "august",
        "sep",
        "sept",
        "september",
        "oct",
        "october",
        "nov",
        "november",
        "dec",
        "december",
    )
)


def _iter_account_raw_texts(raw_lines_path: str) -> Iterable[str]:
    """Yield the text field for each entry in a raw_lines.json file."""

    with open(raw_lines_path, "r", encoding="utf-8") as handle:
        in_object = False
        buffer_lines: list[str] = []

        for line in handle:
            stripped = line.lstrip()

            if not in_object:
                if stripped.startswith("{"):
                    in_object = True
                    buffer_lines = [line]
                continue

            buffer_lines.append(line)

            if stripped.startswith("}") or stripped.startswith("},"):
                in_object = False
                json_blob = "".join(buffer_lines).rstrip().rstrip(",")
                if not json_blob:
                    continue

                try:
                    record = json.loads(json_blob)
                except json.JSONDecodeError:
                    continue

                text = record.get("text")
                if isinstance(text, str):
                    yield text


def _count_english_months(text: str) -> int:
    tokens: list[str] = []
    current: list[str] = []

    for char in text:
        if char.isalpha():
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))

    hits = 0
    for token in tokens:
        if token.lower() in ENGLISH_MONTH_TOKENS:
            hits += 1
    return hits


def _count_hebrew_months(text: str) -> int:
    return sum(text.count(month) for month in HEBREW_MONTHS)


def detect_month_language_for_run(run_dir: str) -> Dict[str, object]:
    """Scans all accounts' raw_lines.json files to infer the month language."""

    accounts_dir = os.path.join(run_dir, "cases", "accounts")

    total_he_hits = 0
    total_en_hits = 0
    accounts_scanned = 0

    if os.path.isdir(accounts_dir):
        for entry in sorted(os.scandir(accounts_dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue

            raw_lines_path = os.path.join(entry.path, "raw_lines.json")
            if not os.path.isfile(raw_lines_path):
                continue

            accounts_scanned += 1
            marker_found = False

            for text in _iter_account_raw_texts(raw_lines_path):
                if not marker_found:
                    if "Two-Year Payment History" in text:
                        marker_found = True
                    else:
                        continue

                total_he_hits += _count_hebrew_months(text)
                total_en_hits += _count_english_months(text)

    if total_he_hits > total_en_hits:
        month_language = "he"
        convention = "DMY"
        confidence = 1.0
    elif total_en_hits > total_he_hits:
        month_language = "en"
        convention = "MDY"
        confidence = 1.0
    else:
        month_language = "unknown"
        convention = None
        confidence = 0.0

    return {
        "date_convention": {
            "scope": "global",
            "convention": convention,
            "month_language": month_language,
            "confidence": confidence,
            "evidence_counts": {
                "he_hits": total_he_hits,
                "en_hits": total_en_hits,
                "accounts_scanned": accounts_scanned,
            },
            "detector_version": "1.0",
        }
    }
