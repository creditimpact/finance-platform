from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterable, Mapping

from . import config as merge_config

WANTED_CONTEXT_KEYS: list[str] = [
    "Account #",
    "High Balance:",
    "Last Verified:",
    "Date of Last Activity:",
    "Date Reported:",
    "Date Opened:",
    "Balance Owed:",
    "Closed Date:",
    "Account Rating:",
    "Account Description:",
    "Dispute Status:",
    "Creditor Type:",
    "Account Status:",
    "Payment Status:",
    "Creditor Remarks:",
    "Payment Amount:",
    "Last Payment:",
    "Past Due Amount:",
    "Account Type:",
    "Credit Limit:",
]

DEFAULT_MAX_LINES = 20
MAX_CONTEXT_LINE_LENGTH = 240


def _coerce_text(entry: object) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping):
        value = entry.get("text")
        if isinstance(value, str):
            return value
        if value is not None:
            return str(value)
    if entry is None:
        return ""
    return str(entry)


def _normalize_line(text: str) -> str:
    norm = text or ""
    norm = norm.replace("\u2013", "-").replace("\u2014", "-")
    norm = re.sub(r"\s+", " ", norm).strip()
    if len(norm) > MAX_CONTEXT_LINE_LENGTH:
        norm = norm[: MAX_CONTEXT_LINE_LENGTH - 3].rstrip() + "..."
    return norm


def _is_only_dashes(text: str) -> bool:
    if not text:
        return True
    return re.sub(r"[-\s]", "", text) == ""


def _should_include_value(text: str, wanted_keys: Iterable[str]) -> bool:
    for key in wanted_keys:
        if key and key in text:
            return True
    return False


def _coerce_positive_int(value: object, default: int) -> int:
    try:
        number = int(str(value))
    except Exception:
        return default
    return number if number > 0 else default


def extract_context_raw(
    raw_lines: list[dict] | list[object],
    wanted_keys: list[str] | None,
    max_lines: int,
) -> list[str]:
    keys = wanted_keys or WANTED_CONTEXT_KEYS
    limit = max_lines if max_lines and max_lines > 0 else DEFAULT_MAX_LINES
    if limit <= 0:
        return []

    original_texts = [_coerce_text(entry) for entry in raw_lines or []]
    normalized = [_normalize_line(text) for text in original_texts]

    interesting_indices: list[int] = []
    for idx, line in enumerate(normalized):
        if not line or _is_only_dashes(line):
            continue
        if _should_include_value(line, keys):
            interesting_indices.append(idx)

    if not interesting_indices:
        return []

    header_index: int | None = None
    first_idx = interesting_indices[0]
    for idx in range(first_idx):
        candidate = normalized[idx]
        if candidate and not _is_only_dashes(candidate):
            header_index = idx
            break

    ordered_indices: list[int] = []
    if header_index is not None:
        ordered_indices.append(header_index)
    for idx in interesting_indices:
        if idx not in ordered_indices:
            ordered_indices.append(idx)

    context: list[str] = []
    seen: set[str] = set()
    for idx in ordered_indices:
        if len(context) >= limit:
            break
        line = normalized[idx]
        if not line or _is_only_dashes(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        context.append(line)

    return context[:limit]


def _load_raw_lines(path: Path) -> list[object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError(f"raw_lines payload must be a list: {path}")


def _extract_account_number(raw_lines: Iterable[str]) -> str | None:
    pattern = re.compile(r"Account #\s*(.*)", re.IGNORECASE)
    for line in raw_lines:
        match = pattern.search(line)
        if not match:
            continue
        tail = match.group(1).strip()
        if not tail:
            continue
        parts = [part.strip(" -:") for part in re.split(r"--", tail)]
        for part in parts:
            if part and not _is_only_dashes(part):
                return part
    return None


def _build_pack_payload(
    sid: str,
    first_idx: int,
    second_idx: int,
    first_context: list[str],
    second_context: list[str],
    first_account_number: str | None,
    second_account_number: str | None,
    highlights: Mapping[str, object] | None,
    max_lines: int,
) -> dict:
    return {
        "sid": sid,
        "pair": {"a": first_idx, "b": second_idx},
        "highlights": dict(highlights or {}),
        "context": {"a": list(first_context), "b": list(second_context)},
        "ids": {
            "account_number_a": first_account_number or "--",
            "account_number_b": second_account_number or "--",
        },
        "limits": {"max_lines_per_side": max_lines},
    }


def _write_pack(path: Path, payload: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    path.write_text(serialized + "\n", encoding="utf-8")


def build_ai_pack_for_pair(
    sid: str,
    runs_root: str | os.PathLike[str],
    a_idx: int,
    b_idx: int,
    highlights: Mapping[str, object] | None,
    *,
    overwrite: bool = False,
) -> dict:
    sid_str = str(sid)
    runs_root_path = Path(runs_root)
    accounts_root = runs_root_path / sid_str / "cases" / "accounts"

    try:
        account_a = int(a_idx)
        account_b = int(b_idx)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Account indices must be integers") from exc

    raw_a_path = accounts_root / str(account_a) / "raw_lines.json"
    raw_b_path = accounts_root / str(account_b) / "raw_lines.json"

    raw_lines_a = _load_raw_lines(raw_a_path)
    raw_lines_b = _load_raw_lines(raw_b_path)

    max_lines = merge_config.get_ai_pack_max_lines_per_side()

    context_a = extract_context_raw(raw_lines_a, WANTED_CONTEXT_KEYS, max_lines)
    context_b = extract_context_raw(raw_lines_b, WANTED_CONTEXT_KEYS, max_lines)

    normalized_a = [_normalize_line(_coerce_text(line)) for line in raw_lines_a or []]
    normalized_b = [_normalize_line(_coerce_text(line)) for line in raw_lines_b or []]

    account_number_a = _extract_account_number(normalized_a)
    account_number_b = _extract_account_number(normalized_b)

    pack_for_a = _build_pack_payload(
        sid_str,
        account_a,
        account_b,
        context_a,
        context_b,
        account_number_a,
        account_number_b,
        highlights,
        max_lines,
    )
    pack_for_b = _build_pack_payload(
        sid_str,
        account_b,
        account_a,
        context_b,
        context_a,
        account_number_b,
        account_number_a,
        highlights,
        max_lines,
    )

    pack_a_path = accounts_root / str(account_a) / "ai" / f"pack_pair_{account_a}_{account_b}.json"
    pack_b_path = accounts_root / str(account_b) / "ai" / f"pack_pair_{account_b}_{account_a}.json"

    _write_pack(pack_a_path, pack_for_a, overwrite)
    _write_pack(pack_b_path, pack_for_b, overwrite)

    return pack_for_a

