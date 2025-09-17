try:  # pragma: no cover - import shim
    import scripts._bootstrap  # KEEP FIRST
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parent.parent
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    import scripts._bootstrap  # type: ignore  # KEEP FIRST

#!/usr/bin/env python3
"""Split accounts from a full token TSV dump.

This script groups tokens by `(page, line)` to form lines of text. It detects
account boundaries based on lines that contain the exact string ``Account #``.
Each account is emitted to a structured JSON file and, optionally, into
individual TSV files for debugging.
"""

import argparse
import csv
import json
import logging
import math
import os
import re
import shutil
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.config import RAW_JOIN_TOKENS_WITH_SPACE, RAW_TRIAD_FROM_X
from backend.core.logic.report_analysis.canonical_labels import LABEL_MAP
from backend.core.logic.report_analysis.normalize_fields import (
    clean_value,
    ensure_all_keys,
)
from backend.core.logic.report_analysis.triad_layout import (
    TRIAD_BOUNDARY_GUARD,
    TriadLayout,
    assign_band,
    bands_from_header_tokens,
)
from backend.pipeline.runs import RunManifest, write_breadcrumb

logger = logging.getLogger(__name__)
# Enable with RAW_TRIAD_FROM_X=1 for verbose triad logs
triad_log = logger.info if RAW_TRIAD_FROM_X else (lambda *a, **k: None)
TRIAD_BAND_BY_X0 = os.environ.get("TRIAD_BAND_BY_X0") == "1"


def _env_truthy_opt(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() not in {"0", "false", "no"}


def _resolve_split_space_triads() -> bool:
    stagea_flag = _env_truthy_opt("STAGEA_SPLIT_SPACE_TRIADS")
    if stagea_flag is not None:
        return stagea_flag
    legacy_flag = _env_truthy_opt("SPLIT_SPACE_TRIADS")
    if legacy_flag is not None:
        return legacy_flag
    return True


SPLIT_SPACE_TRIADS = _resolve_split_space_triads()
STAGEA_DEBUG = os.environ.get("STAGEA_DEBUG") == "1"


def _stagea_debug(message: str, *args) -> None:
    if STAGEA_DEBUG:
        logger.info(message, *args)


def _dbg_path(label: str, p: Path | str | None):
    """Best-effort path resolver debug helper.

    Logs a PATH_RESOLVE line that shows which code path supplied a value and
    what it resolves to on disk. Returns the resolved/printed string.
    """
    try:
        rp = str(Path(p).resolve()) if p is not None else "<None>"
    except Exception:
        try:
            rp = str(p)  # type: ignore[arg-type]
        except Exception:
            rp = "<unprintable>"
    logger.info("PATH_RESOLVE %s=%s", label, rp)
    return rp


# Tunables for x0 mode
def _get_float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


TRIAD_X0_TOL: float = _get_float_env("TRIAD_X0_TOL", 0.5)
TRIAD_CONT_NEAREST_MAXDX: float = _get_float_env("TRIAD_CONT_NEAREST_MAXDX", 30.0)

TRACE_ON = (
    os.getenv("TRIAD_TRACE_CSV", "0") == "1"
    and os.getenv("KEEP_PER_ACCOUNT_TSV", "0") == "1"
)
trace_dir: Path | None = None
_trace_fp = None
_trace_wr = None


def _trace_open(path: Path | str) -> None:
    """Open a per-account trace CSV under ``trace_dir`` with required header."""
    global _trace_fp, _trace_wr
    if not TRACE_ON or trace_dir is None:
        return
    try:
        if _trace_fp:
            _trace_fp.close()
    except Exception:
        pass
    p = trace_dir / Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _trace_fp = open(p, "w", newline="", encoding="utf-8")
    _trace_wr = csv.writer(_trace_fp)
    _trace_wr.writerow(
        [
            "page",
            "line",
            "token",
            "text",
            "x0",
            "x1",
            "mid_x",
            "band",
            "phase",
            "label_key",
            "used_axis",
            "reassigned_from",
            "wrap_affinity",
        ]
    )


def _trace(page, line, t, band, action):
    """Legacy trace writer (kept for compatibility)."""
    if _trace_wr:
        try:
            mid = (float(t.get("x0", 0)) + float(t.get("x1", 0))) / 2.0
        except Exception:
            mid = 0.0
        _trace_wr.writerow(
            [
                page,
                line,
                "",
                t.get("text"),
                t.get("x0"),
                t.get("x1"),
                mid,
                band,
                action,
                "",
                ("x0" if TRIAD_BAND_BY_X0 else "mid"),
                "",
                "",
            ]
        )


def _trace_token(
    page,
    line,
    token_index,
    t,
    band,
    phase,
    label_key: str | None = None,
    reassigned_from: str = "",
    wrap_affinity: str = "",
):
    if _trace_wr:
        try:
            mid = (float(t.get("x0", 0)) + float(t.get("x1", 0))) / 2.0
        except Exception:
            mid = 0.0
        _trace_wr.writerow(
            [
                page,
                line,
                token_index,
                t.get("text"),
                t.get("x0"),
                t.get("x1"),
                mid,
                band,
                phase,
                label_key or "",
                ("x0" if TRIAD_BAND_BY_X0 else "mid"),
                reassigned_from,
                wrap_affinity,
            ]
        )


def _triad_band_log(message: str, *args) -> None:
    """Emit detailed band diagnostics when trace/debugging is enabled."""

    if TRACE_ON or STAGEA_DEBUG or logger.isEnabledFor(logging.DEBUG):
        logger.info(message, *args)


_triad_x0_fallback_logged: set[int] = set()

STOP_MARKER_NORM = "publicinformation"
SECTION_STARTERS = {"collection", "unknown"}
_SECTION_NAME = {"collection": "collections", "unknown": "unknown"}
NOISE_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
NOISE_BANNER_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4}.*(?:Credit\s*Report|SmartCredit)", re.IGNORECASE
)

# How many lines above the ``Account #`` anchor to consider the heading.
# Per hardening rules we look farther back to find a suitable headline.
HEADING_BACK_LINES = 8


_SPACE_RE = re.compile(r"\s+")
MULTISPACE = re.compile(r"\s{2,}")
DOT_DATE_RE = re.compile(r"\d{1,2}\.\d{1,2}\.\d{2,4}")
SLASH_DATE_RE = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}/\d{2,4}")
MONTH_YEAR_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{2,4}\b",
    re.IGNORECASE,
)
NUMERIC_TOKEN_RE = re.compile(r"\b\d[\d,./]*\b")
MONTH_TOKEN_RE = re.compile(
    r"^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?$",
    re.IGNORECASE,
)
YEAR_TOKEN_RE = re.compile(r"^\d{2,4}$")
MONEY_RE = re.compile(r"\$\s*\d")


def join_tokens_with_space(tokens: Iterable[str]) -> str:
    """Join tokens with a single space, normalizing whitespace."""
    s = " ".join(t.strip() for t in tokens if t is not None)
    return _SPACE_RE.sub(" ", s).strip()


def _norm_simple(text: str) -> str:
    """Legacy normalization used for structural guards.

    Removes non-alphanumeric characters and casefolds the result.
    """
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _norm(s: str) -> str:
    """Normalize ``s`` by dropping ``\N{REGISTERED SIGN}`` and collapsing whitespace."""
    return re.sub(r"\s+", " ", s.replace("\u00ae", "")).strip()


def _clean_value(txt: str) -> str:
    s = _norm(txt)
    if not s:
        return ""
    return clean_value(s)


def _split_triad_tail_3cols(tail: str) -> List[str]:
    """Split a triad tail (after a label) into three bureau segments."""

    s = (tail or "").replace("\u00a0", " ").strip()
    if not s:
        return ["", "", ""]

    parts: List[str]
    normalized = s.replace("—", "--").replace("–", "--")
    compact = normalized.replace(" ", "")
    if compact and set(compact) <= {"-"}:
        return ["--", "--", "--"]
    has_dash_delim = " -- " in normalized or normalized.count("--") >= 2
    if has_dash_delim:
        raw_parts = [p.strip() for p in normalized.split("--")]
        parts = [p if p else "--" for p in raw_parts]
    else:
        spaced = [p.strip() for p in MULTISPACE.split(normalized) if p.strip()]
        if len(spaced) >= 3:
            parts = spaced
        else:
            tokens = [tok for tok in normalized.split(" ") if tok]
            if len(tokens) >= 3:
                parts = _group_single_space_tokens(tokens)
            else:
                parts = tokens

    return _normalize_triad_parts(parts)


def _group_single_space_tokens(tokens: List[str]) -> List[str]:
    """Group single-space-delimited tokens into up to three value segments."""

    grouped: List[str] = []
    idx = 0
    # Capture up to the first two segments explicitly, remainder goes to the tail
    while idx < len(tokens) and len(grouped) < 2:
        tok = tokens[idx]
        take_tokens = [tok]
        next_idx = idx + 1
        cleaned = re.sub(r"[.,]$", "", tok)
        if MONTH_TOKEN_RE.match(cleaned) and next_idx < len(tokens):
            nxt = tokens[next_idx]
            nxt_clean = re.sub(r"[.,]$", "", nxt)
            if YEAR_TOKEN_RE.match(nxt_clean):
                take_tokens.append(nxt)
                next_idx += 1
        grouped.append(" ".join(take_tokens))
        idx = next_idx

    remainder = " ".join(tokens[idx:]) if idx < len(tokens) else ""
    grouped.append(remainder)
    return grouped


def _normalize_triad_parts(parts: List[str]) -> List[str]:
    cleaned = [_clean_value(p) for p in parts]
    while len(cleaned) < 3:
        cleaned.append("")
    while len(cleaned) > 3:
        merged = f"{cleaned[-2]} {cleaned[-1]}".strip()
        cleaned[-2] = _clean_value(merged)
        cleaned.pop()
    return [c or "" for c in cleaned]


_TRIAD_PRE_SPLIT_BANDS: List[Tuple[str, str]] = [
    ("tu", "transunion"),
    ("xp", "experian"),
    ("eq", "equifax"),
]


def _band_window_for_synthetic(layout: TriadLayout, band: str) -> Tuple[float, float]:
    if band == "tu":
        left, right = layout.tu_band
    elif band == "xp":
        left, right = layout.xp_band
    else:
        left = layout.eq_band[0]
        xp_width = layout.xp_band[1] - layout.xp_band[0]
        if not (xp_width > 0):
            xp_width = layout.tu_band[1] - layout.tu_band[0]
        if not (xp_width > 0):
            xp_width = 80.0
        right = left + xp_width
    if not (right > left):
        right = left + 40.0
    return float(left), float(right)


def _make_synthetic_token(
    token: dict, band: str, text: str, layout: TriadLayout
) -> dict:
    clone = dict(token)
    clone["text"] = text
    left, right = _band_window_for_synthetic(layout, band)
    span = right - left
    if not (span > 0):
        span = 40.0
    width = min(max(span * 0.6, 6.0), 60.0)
    x0 = left + max(2.0, (span - width) / 2.0)
    x1 = x0 + width
    if x1 > right - 0.5:
        x1 = right - 0.5
        x0 = max(left + 0.5, x1 - width)
    if x1 <= x0:
        x0 = left + 1.0
        x1 = x0 + max(1.0, width)
    clone["x0"] = f"{x0:.2f}"
    clone["x1"] = f"{x1:.2f}"
    clone["_synthetic_pre_split"] = "1"
    clone["_synthetic_band"] = band
    return clone


def _pre_split_bureau_token(
    token: dict, layout: TriadLayout
) -> List[Tuple[str, dict, str]] | None:
    if not SPLIT_SPACE_TRIADS:
        return None
    if token.get("_synthetic_pre_split") == "1":
        return None
    parts = _maybe_split_triad_tail(token.get("text"))
    if not parts:
        return None
    payload: List[Tuple[str, dict, str]] = []
    for (band_key, _bureau_name), part in zip(_TRIAD_PRE_SPLIT_BANDS, parts):
        synthetic = _make_synthetic_token(token, band_key, part, layout)
        payload.append((band_key, synthetic, part))
    return payload


def _maybe_split_triad_tail(raw_text: str | None) -> List[str] | None:
    if raw_text is None:
        return None

    sample = str(raw_text).replace("\u00a0", " ").strip()
    if not sample:
        return None

    parts = _split_triad_tail_3cols(sample)
    if not parts:
        return None

    normalized_sample = " ".join(sample.split())
    dash_hits = normalized_sample.count("--")
    looks_multi = dash_hits >= 2
    if not looks_multi and MULTISPACE.search(sample):
        looks_multi = True
    if not looks_multi and len(DOT_DATE_RE.findall(sample)) >= 2:
        looks_multi = True
    if not looks_multi and len(SLASH_DATE_RE.findall(sample)) >= 2:
        looks_multi = True
    if not looks_multi and len(MONTH_YEAR_RE.findall(sample)) >= 2:
        looks_multi = True
    if not looks_multi and len(MONEY_RE.findall(sample)) >= 2:
        looks_multi = True
    if not looks_multi and len(NUMERIC_TOKEN_RE.findall(sample)) >= 3:
        looks_multi = True
    if not looks_multi and all(p == "--" for p in parts):
        looks_multi = dash_hits >= 2

    if not looks_multi:
        return None

    if len(parts) == 3 and parts[0] == normalized_sample and parts[1] == parts[2] == "--":
        return None

    return parts


def _norm_text(s: str) -> str:
    """Normalize text for guard checks by collapsing whitespace and punctuation."""
    s = s.replace("\u00ae", " ")
    s = s.replace(",", " ").replace(":", " ")
    return " ".join(s.split()).casefold()


def _bare_bureau_norm(s: str) -> str:
    return "".join(s.casefold().replace("\u00ae", "").split())


BARE_BUREAUS = {"transunion", "experian", "equifax"}

H2Y_PAT = re.compile(r"\bTwo[-\s]?Year\b.*\bPayment\b.*\bHistory\b", re.I)
H2Y_STATUS_RE = re.compile(r"^(?:ok|co|[0-9]{2,3})$", re.I)
H7Y_TITLE_PAT = re.compile(r"(Days\s*Late|7\s*Year\s*History)", re.I)
H7Y_BUREAUS = ("Transunion", "Experian", "Equifax")
# Recognize "30:", "60:", "90:" optionally followed by a value (digits or --)
# Examples: "30:", "30: 3", "30:3", "30: --"
LATE_KEY_PAT = re.compile(r"^\s*(30|60|90)\s*:\s*(?:(\d+)|--)?\s*$", re.I)

# Plain integer used when the value is on the next token
INT_PAT = re.compile(r"^\d+$")
DASH_PAT = re.compile(r"^--$", re.ASCII)
H7Y_EPS = 6.0


def _header_norm(s: str) -> str:
    """Normalize header text: drop \N{REGISTERED SIGN}, collapse whitespace."""
    return re.sub(r"\s+", " ", s.replace("\u00ae", "")).strip()


def _bureau_key(text_norm: str) -> Optional[str]:
    """Return compact bureau key (tu/xp/eq) for ``text_norm`` if matched."""
    t = text_norm.casefold()
    if t.startswith("transunion"):
        return "tu"
    if t.startswith("experian"):
        return "xp"
    if t.startswith("equifax"):
        return "eq"
    return None


def _slab_of(
    x: float | None, slabs: Optional[Dict[str, Tuple[float, float]]]
) -> Optional[str]:
    """Return bureau key whose slab contains ``x``."""
    if slabs is None or x is None:
        return None
    for k, (a, b) in slabs.items():
        if a <= x < b:
            return k
    return None


def _process_h7y_token(
    t: dict,
    slabs: Dict[str, Tuple[float, float]] | None,
    acc_seven_year: Dict[str, Dict[str, int]],
    last_key: Dict[str, Optional[str]],
) -> None:
    """Process a token within the 7-year history block.

    Assigns counts to ``acc_seven_year`` based on the token's horizontal
    position and updates ``last_key``. Also emits trace rows and debug logs.
    """
    if slabs is None:
        return
    raw = str(t.get("text", ""))
    clean = _clean_value(raw)
    mx: float | None
    try:
        mx = _triad_mid_x(t)
    except Exception:
        mx = None
    if mx is None:
        try:
            mx = float(t.get("x0"))
        except Exception:
            mx = None
    b = _slab_of(mx, slabs)
    if not b:
        return

    m = LATE_KEY_PAT.match(raw)
    if m:
        key = m.group(1)
        inline_val = m.group(2)
        _history_trace(
            t.get("page"),
            t.get("line"),
            phase="history7y",
            text=raw,
            kind=f"key:{key}",
            x0=t.get("x0"),
            x1=t.get("x1"),
        )
        logger.info("H7Y_KEY bureau=%s key=%s", b.upper(), key)
        if inline_val is not None:
            v = int(inline_val)
            acc_seven_year[b][f"late{key}"] = v
            _history_trace(
                t.get("page"),
                t.get("line"),
                phase="history7y",
                kind=f"late{key}",
                value=v,
                x0=t.get("x0"),
                x1=t.get("x1"),
            )
            logger.info(
                "H7Y_VALUE bureau=%s kind=late%s value=%d",
                b.upper(),
                key,
                v,
            )
            last_key[b] = None
        else:
            last_key[b] = key
        return

    if last_key[b]:
        key = last_key[b]
        if INT_PAT.match(raw):
            v = int(raw)
            acc_seven_year[b][f"late{key}"] = v
            _history_trace(
                t.get("page"),
                t.get("line"),
                phase="history7y",
                kind=f"late{key}",
                value=v,
                x0=t.get("x0"),
                x1=t.get("x1"),
            )
            logger.info(
                "H7Y_VALUE bureau=%s kind=late%s value=%d",
                b.upper(),
                key,
                v,
            )
            last_key[b] = None
            return
        if DASH_PAT.match(clean):
            acc_seven_year[b][f"late{key}"] = 0
            _history_trace(
                t.get("page"),
                t.get("line"),
                phase="history7y",
                kind=f"late{key}",
                value=0,
                x0=t.get("x0"),
                x1=t.get("x1"),
            )
            logger.info(
                "H7Y_VALUE bureau=%s kind=late%s value=%d",
                b.upper(),
                key,
                0,
            )
            last_key[b] = None


def _flush_history(account: Optional[dict], acc_two_year, acc_seven_year) -> None:
    """Attach buffered history to ``account`` if provided."""
    if account is None:
        return
    account["two_year_payment_history"] = {
        "transunion": acc_two_year.get("tu", []),
        "experian": acc_two_year.get("xp", []),
        "equifax": acc_two_year.get("eq", []),
    }

    def _seven(b: str) -> Dict[str, int]:
        src = acc_seven_year.get(b, {})
        return {k: int(src.get(k, 0)) for k in ("late30", "late60", "late90")}

    account["seven_year_history"] = {
        "transunion": _seven("tu"),
        "experian": _seven("xp"),
        "equifax": _seven("eq"),
    }


def _trace_write(
    *,
    phase: str,
    bureau: str,
    page: int,
    line: int,
    text: str = "",
    kind: str = "",
    value: int | None = None,
    x0: float | None = None,
    x1: float | None = None,
    mid_x: float | None = None,
) -> None:
    """Generic trace writer for history observers."""
    if not TRACE_ON or not _trace_wr:
        return
    txt = text if text else ("" if value is None else str(value))
    _trace_wr.writerow(
        [
            page,
            line,
            "",
            txt,
            x0,
            x1,
            mid_x,
            bureau.upper(),
            phase,
            kind,
            ("x0" if TRIAD_BAND_BY_X0 else "mid"),
            "",
            "",
        ]
    )


def _history_trace(
    page: int,
    line: int,
    phase: str,
    text: str = "",
    kind: str = "",
    value: int | None = None,
    x0: float | None = None,
    x1: float | None = None,
) -> None:
    """Generic trace row for history observers (2Y/7Y)."""
    if not TRACE_ON or not _trace_wr:
        return
    txt = text if text else ("" if value is None else str(value))
    _trace_wr.writerow(
        [
            page,
            line,
            "",
            txt,
            x0,
            x1,
            phase,
            kind,
            "",
        ]
    )


def _is_triad(text: str) -> bool:
    """Return True if ``text`` is the TransUnion/Experian/Equifax triad.

    The match is flexible and ignores punctuation such as ®, commas or
    colons. The three bureau names may appear in any order. When a match is
    detected a debug log is emitted for traceability.
    """
    s = (text or "").lower()
    s = s.replace("\u00ae", "")
    s = s.replace(",", " ").replace(":", " ")
    s = re.sub(r"\s+", " ", s).strip()
    hit = all(name in s for name in ("transunion", "experian", "equifax"))
    if hit:
        triad_log("TRIAD_HEADER_MATCH raw=%r norm=%r", text, s)
    return hit


def _is_anchor(text: str) -> bool:
    """Return True if ``text`` contains the literal ``Account #`` anchor."""
    return "account#" in _norm_simple(text)


def is_account_anchor(joined_text: str) -> bool:
    """Return True if ``joined_text`` matches the exact ``Account #`` anchor."""
    # "Account" without the trailing ``#`` must not trigger activation
    return joined_text.strip().startswith("Account #")


def _looks_like_headline(text: str) -> bool:
    """Return True if ``text`` is an ALL-CAPS headline candidate."""
    stripped = re.sub(r"[^A-Za-z0-9/&\- ]", "", text).strip()
    if ":" in stripped:
        return False
    core = stripped.replace(" ", "")
    if len(core) < 3:
        return False
    return core.isupper()


def _mid_y(t: dict) -> float:
    """Return the vertical midpoint of token ``t``."""
    try:
        y0 = float(t.get("y0", 0.0))
        y1 = float(t.get("y1", y0))
        return (y0 + y1) / 2.0
    except Exception:
        return 0.0


UNICODE_COLONS = "\u003a\uff1a\ufe55\ufe13"  # ":" ， "：" ， "﹕" ， "︓"


GRID_TOKENS = {"ok", "0", "30", "60", "90"}


def _is_history_grid_line(banded_tokens: Dict[str, List[dict]]) -> bool:
    """Return True if tokens across all bureaus form a payment history grid."""

    def _col_val(toks: List[dict]) -> str | None:
        if not toks:
            return None
        s = "".join(t.get("text", "") for t in toks)
        s = re.sub(r"\s+", "", s).lower()
        return s

    return (
        _col_val(banded_tokens.get("tu", [])) in GRID_TOKENS
        and _col_val(banded_tokens.get("xp", [])) in GRID_TOKENS
        and _col_val(banded_tokens.get("eq", [])) in GRID_TOKENS
    )


def _has_label_suffix(txt: str) -> bool:
    s = (txt or "").strip()
    return s.endswith(("#",) + UNICODE_COLONS_TUP)


def _assign_band_any(
    token: dict, layout: TriadLayout, row_eq_right_x0: float | None = None
) -> str:
    if TRIAD_BAND_BY_X0:
        try:
            x0 = float(token.get("x0", 0.0))
        except Exception:
            x0 = 0.0

        tol = TRIAD_X0_TOL
        guard = TRIAD_BOUNDARY_GUARD

        tu_left = float(layout.tu_left_x0 or layout.tu_band[0] or 0.0)
        xp_left = float(layout.xp_left_x0 or layout.xp_band[0] or 0.0)
        eq_left = float(layout.eq_left_x0 or layout.eq_band[0] or 0.0)

        eq_band_right = getattr(layout, "eq_band", (0.0, float("inf")))[1]
        try:
            eq_band_right = float(eq_band_right)
        except Exception:
            eq_band_right = float("inf")

        eq_right = row_eq_right_x0 if row_eq_right_x0 is not None else eq_band_right
        if not math.isfinite(eq_right):
            eq_right = float("inf")
        if eq_right < eq_left:
            eq_right = eq_left

        text = str(token.get("text", ""))
        if _is_label_token_text(text) and x0 <= (tu_left + guard):
            return "label"

        if x0 < (tu_left - tol):
            return "label"

        band = "none"
        if x0 < xp_left:
            band = "tu"
        elif x0 < eq_left:
            band = "xp"
        elif x0 < eq_right:
            band = "eq"

        if band == "xp" and xp_left and xp_left < x0 < (xp_left + guard):
            return "tu"
        if band == "eq":
            if eq_left and eq_left < x0 < (eq_left + guard):
                return "xp"
            if row_eq_right_x0 is not None and x0 >= eq_right:
                return "none"
            return "eq"

        return band
    band = assign_band(token, layout)
    if band == "tu":
        try:
            tu_left = float(layout.tu_band[0])
            xp_left = float(layout.xp_band[0])
        except Exception:
            tu_left = xp_left = None
        if xp_left is not None and tu_left is not None and xp_left > tu_left:
            seam = (tu_left + xp_left) / 2.0
            if _triad_mid_x(token) >= seam:
                return "xp"
    if band == "xp":
        try:
            xp_left = float(layout.xp_band[0])
            eq_left = float(layout.eq_band[0])
        except Exception:
            xp_left = eq_left = None
        if eq_left is not None and xp_left is not None and eq_left > xp_left:
            seam = (xp_left + eq_left) / 2.0
            if _triad_mid_x(token) >= seam:
                return "eq"
    return band


def _band_mid_and_eps(band: Tuple[float, float]) -> tuple[float, float]:
    left, right = band
    try:
        left_f = float(left)
    except Exception:
        left_f = 0.0
    try:
        right_f = float(right)
    except Exception:
        right_f = float("inf")
    width = right_f - left_f
    if not math.isfinite(width) or width <= 0:
        width = 160.0
    eps = max(10.0, min(80.0, width / 2.0))
    mid = left_f + eps
    return mid, eps


def in_label_band(token: dict, layout: TriadLayout) -> bool:
    """Return True if ``token`` lies within the label band of ``layout``."""
    band = _assign_band_any(token, layout)
    _trace(token.get("page"), token.get("line"), token, band, "in_label_band")
    return band == "label"


def verify_anchor_row(tokens: List[dict], layout: TriadLayout) -> bool:
    """Legacy strict validator for anchor rows (kept for compatibility).

    Note: Triad activation now uses the relaxed `_validate_anchor_row` below.
    """
    bands: Dict[str, List[dict]] = {"label": [], "tu": [], "xp": [], "eq": []}
    for t in tokens:
        b = _assign_band_any(t, layout)
        _trace(t.get("page"), t.get("line"), t, b, "verify_anchor_row")
        if b in bands:
            bands[b].append(t)
    if not bands["label"] or not all(len(bands[b]) == 1 for b in ("tu", "xp", "eq")):
        return False
    label_norms = [_norm_text(t.get("text", "")) for t in bands["label"]]
    if not any("account" in n for n in label_norms):
        return False
    names = {"tu": "transunion", "xp": "experian", "eq": "equifax"}
    for band, name in names.items():
        txt = _norm_text(bands[band][0].get("text", ""))
        if name not in txt:
            return False
    return True


# --- Anchor validation: accept >=1 token per bureau ---
NOISE_TOKENS = {"-", "—", "–", "|"}


def _triad_mid_x(t: dict) -> float:
    try:
        x0 = float(t.get("x0", 0.0))
        x1 = float(t.get("x1", x0))
        return (x0 + x1) / 2.0
    except Exception:
        return 0.0


def _token_band(
    t: dict, layout: TriadLayout, row_eq_right_x0: float | None = None
) -> str:
    # Use local helper; assign by geometry only
    return _assign_band_any(t, layout, row_eq_right_x0=row_eq_right_x0)


def _validate_anchor_row(
    anchor_tokens: List[dict], layout: TriadLayout
) -> tuple[bool, Dict[str, int]]:
    """Relaxed Account # anchor validator.

    Returns a tuple ``(has_label, counts)`` where ``counts`` maps each band to
    the number of non-noise tokens observed. A valid anchor must have a label
    token in the label band (ending with '#', ':' or Unicode colon variants).
    Purely geometry-based; ignores content heuristics. Also logs band counts for
    diagnostics.
    """

    by_band: Dict[str, int] = {"label": 0, "tu": 0, "xp": 0, "eq": 0}
    for idx, t in enumerate(anchor_tokens):
        txt = str(t.get("text", "")).strip()
        if txt in NOISE_TOKENS:
            continue
        b = _token_band(t, layout)
        _trace(t.get("page"), t.get("line"), t, b, "validate_anchor_row_relaxed")
        _trace_token(
            t.get("page"), t.get("line"), idx, t, b, "anchor", "account_number_display"
        )
        if b in by_band:
            by_band[b] += 1

    # Label must have a trailing marker and be in the label band.
    # Snap tolerance: if first token sits within 2*eps left of the TU midpoint,
    # accept it as a label even if it barely crosses the seam.
    label_texts = [
        str(t.get("text", ""))
        for t in anchor_tokens
        if _token_band(t, layout) == "label"
    ]
    has_label = any(
        s.strip().endswith(("#",) + UNICODE_COLONS_TUP) for s in label_texts
    )

    if has_label:
        try:
            label_xs = [
                float(t.get("x0", 0.0))
                for t in anchor_tokens
                if _token_band(t, layout) == "label"
            ]
        except Exception:
            label_xs = []
        try:
            label_right = float(layout.label_band[1])
        except Exception:
            label_right = 0.0
        if label_xs and label_right and min(label_xs) >= label_right:
            has_label = False
            by_band["label"] = 0
            logger.info("TRIAD_X0_FALLBACK_OK anchor")

    if not has_label and anchor_tokens:
        first = anchor_tokens[0]
        try:
            x0 = float(first.get("x0", 0.0))
            x1 = float(first.get("x1", x0))
            mid = (x0 + x1) / 2.0
        except Exception:
            mid = 0.0
        # Estimate TU midpoint from layout using band width heuristics
        tu_mid_est, eps = _band_mid_and_eps(layout.tu_band)
        if (tu_mid_est - 2 * eps) <= mid < tu_mid_est:
            if str(first.get("text", "")).strip().endswith(("#",) + UNICODE_COLONS_TUP):
                has_label = True

    logger.info(
        "TRIAD_ANCHOR_COUNTS label=%d tu=%d xp=%d eq=%d",
        by_band["label"],
        by_band["tu"],
        by_band["xp"],
        by_band["eq"],
    )

    # Accept anchors with a label even if bureau tokens are on the next line
    return has_label, by_band


# --- Labeled row processing: split label, then band values per bureau ---
UNICODE_COLONS_TUP = (":", "：", "﹕", "︓")


def _is_label_token_text(txt: str) -> bool:
    s = (txt or "").strip()
    return s.endswith(("#",) + UNICODE_COLONS_TUP)


def _strip_label_suffix(txt: str) -> str:
    s = (txt or "").rstrip()
    for ch in UNICODE_COLONS_TUP:
        if s.endswith(ch):
            return s[:-1].rstrip()
    if s.endswith("#"):
        return s[:-1].rstrip()
    return s


def _strip_colon_only(txt: str) -> str:
    s = (txt or "").rstrip()
    for ch in UNICODE_COLONS_TUP:
        if s.endswith(ch):
            return s[:-1].rstrip()
    return s


def normalize_label_text(s: str) -> str:
    """Normalize visual label text for canonical LABEL_MAP lookup.

    - Preserve '#'
    - Normalize NBSP/thin spaces to regular spaces
    - Normalize en/em dashes to '-'
    - Strip trailing colon variants only (keep '#')
    - Collapse internal whitespace
    """
    s0 = (
        (s or "")
        .replace("\u00a0", " ")
        .replace("\u2009", " ")
        .replace("\u202f", " ")
        .strip()
    )
    s0 = s0.replace("–", "-").replace("—", "-")
    # Manually strip unicode colons, but keep '#'
    for ch in UNICODE_COLONS_TUP:
        if s0.endswith(ch):
            s0 = s0[: -len(ch)].rstrip()
            break
    return " ".join(s0.split())


def process_triad_labeled_line(
    tokens: List[dict],
    layout: TriadLayout,
    label_map: Dict[str, str],
    open_row: Dict[str, Any] | None,
    triad_fields: Dict[str, Dict[str, str]],
    triad_order: List[str],
    account_index: int | None = None,
    pre_split_override: Sequence[str] | None = None,
    pre_split_override_text: str | None = None,
):
    """
    Process a labeled triad line using geometry-only banding.

    Returns None to indicate a layout mismatch that should stop triad;
    otherwise returns the new/open row dict to persist.
    """
    # 1) Build label from multiple tokens: collect label-band tokens from start
    # up to and including the first suffix token (one of '#', ASCII/Unicode colons)
    suffixes = ("#",) + UNICODE_COLONS_TUP
    label_span: List[dict] = []
    suffix_idx: int | None = None
    suffix_was_captured: bool = False

    def _looks_like_value_text(s: str) -> bool:
        z = (s or "").strip()
        if not z:
            return False
        if z.startswith("$"):
            return True
        if z in {"--", "—", "–"}:
            return True
        return bool(re.match(r"^[0-9][0-9,]*(?:\.[0-9]+)?$", z))

    for i, t in enumerate(tokens):
        if _token_band(t, layout) != "label":
            continue
        txt = str(t.get("text", ""))
        # Stop collecting once a value-looking token is seen; don't swallow values into label
        # In x0 mode: stop before TU left edge to avoid swallowing values
        if TRIAD_BAND_BY_X0:
            try:
                x0 = float(t.get("x0", 0.0))
            except Exception:
                x0 = 0.0
            try:
                tu_left_x0 = float(getattr(layout, "tu_left_x0", 0.0))
            except Exception:
                tu_left_x0 = 0.0
            # Stop label collection once a label-band token reaches the TU cutoff (with tolerance)
            if tu_left_x0 and (x0 + TRIAD_X0_TOL) >= tu_left_x0:
                if label_span or not _is_label_token_text(txt):
                    if label_span:
                        suffix_idx = i - 1
                    logger.info(
                        "TRIAD_LABEL_STOP reason=hit_tu_left_x0 x0=%.1f tu_left_x0=%.1f",
                        x0,
                        tu_left_x0,
                    )
                    break
        if _looks_like_value_text(txt):
            # set suffix position to last label token collected so far
            if label_span:
                suffix_idx = i - 1
            logger.info("TRIAD_LABEL_STOP reason=value_token token=%r", txt)
        label_span.append(t)
        if txt.strip().endswith(suffixes):
            suffix_idx = i
            suffix_was_captured = True
            break

    if not label_span:
        logger.info("TRIAD_STOP reason=layout_mismatch_label_band")
        return None
    # If no explicit suffix token found, treat the last collected label token as the split point
    if suffix_idx is None:
        # suffix_idx should point at the last label token index in the original tokens list
        last = label_span[-1]
        try:
            suffix_idx = tokens.index(last)
        except ValueError:
            suffix_idx = 0

    tail_tokens = tokens[suffix_idx + 1 :]
    row_eq_right_x0: float | None = None
    try:
        eq_left_x0_for_log = float(getattr(layout, "eq_left_x0", layout.eq_band[0]))
    except Exception:
        eq_left_x0_for_log = float(layout.eq_band[0])
    if TRIAD_BAND_BY_X0:
        if tail_tokens:
            for next_label_token in tail_tokens:
                text = str(next_label_token.get("text", ""))
                if not _is_label_token_text(text):
                    continue
                try:
                    candidate_x0 = float(next_label_token.get("x0", 0.0))
                except Exception:
                    continue
                if (
                    eq_left_x0_for_log
                    and candidate_x0 + TRIAD_X0_TOL < eq_left_x0_for_log
                ):
                    continue
                row_eq_right_x0 = candidate_x0
                break
        if (
            row_eq_right_x0 is not None
            and eq_left_x0_for_log
            and row_eq_right_x0 < eq_left_x0_for_log
        ):
            row_eq_right_x0 = eq_left_x0_for_log

    visu_label = " ".join(
        (str(t.get("text", "")) or "").strip() for t in label_span
    ).strip()
    canon_label = normalize_label_text(visu_label)
    canonical = label_map.get(canon_label)
    logger.info(
        "TRIAD_LABEL_BUILT visu=%r canon=%r key=%r", visu_label, canon_label, canonical
    )

    label_for_log = canonical or canon_label
    try:
        label_left = float(layout.label_band[0])
    except Exception:
        label_left = 0.0
    try:
        label_right = float(layout.label_band[1])
    except Exception:
        label_right = 0.0
    try:
        tu_left = float(layout.tu_band[0])
    except Exception:
        tu_left = 0.0
    try:
        tu_right = float(layout.tu_band[1])
    except Exception:
        tu_right = 0.0
    try:
        xp_left = float(layout.xp_band[0])
    except Exception:
        xp_left = 0.0
    try:
        xp_right = float(layout.xp_band[1])
    except Exception:
        xp_right = 0.0
    eq_left = eq_left_x0_for_log
    try:
        eq_band_right = float(layout.eq_band[1])
    except Exception:
        eq_band_right = float("inf")
    eq_right_for_log = (
        row_eq_right_x0 if row_eq_right_x0 is not None else eq_band_right
    )
    guard = TRIAD_BOUNDARY_GUARD
    tu_right_guarded = max(tu_left, xp_left - guard)
    xp_right_guarded = max(xp_left, eq_left - guard)
    if not math.isfinite(eq_right_for_log):
        eq_right_guarded = eq_right_for_log
    else:
        eq_right_guarded = max(eq_left, eq_right_for_log - guard)
    _triad_band_log(
        "ROW_BANDS key=%s guard=%.2f headers=(tu=%.3f,xp=%.3f,eq=%.3f) "
        "label=[%.3f,%.3f) tu=[%.3f,%.3f) xp=[%.3f,%.3f) eq=[%.3f,%.3f)",
        label_for_log,
        guard,
        tu_left,
        xp_left,
        eq_left,
        label_left,
        label_right,
        tu_left,
        tu_right_guarded,
        xp_left,
        xp_right_guarded,
        eq_left,
        eq_right_guarded,
    )

    # Task 5: Strict line-break rule — if no suffix captured and the last label token
    # is still left of TU's left x0 cutoff, expect values to start on the next line.
    expect_values_on_next_line = False
    if TRIAD_BAND_BY_X0 and (not suffix_was_captured):
        try:
            last_x0 = float(label_span[-1].get("x0", 0.0))
        except Exception:
            last_x0 = 0.0
        try:
            tu_left_x0 = float(getattr(layout, "tu_left_x0", 0.0))
        except Exception:
            tu_left_x0 = 0.0
        # Only expect continuation if the last label token is clearly left of TU cutoff
        if tu_left_x0 and ((last_x0 + TRIAD_X0_TOL) < tu_left_x0):
            expect_values_on_next_line = True
            logger.info(
                "TRIAD_LABEL_LINEBREAK key=%s -> expecting values on next line",
                canonical,
            )

    def _int_for_log(raw: Any) -> Any:
        try:
            return int(float(raw))
        except Exception:
            return "?"

    def _float_for_log(raw: Any) -> float:
        try:
            return float(raw)
        except Exception:
            return 0.0

    dash_tokens = {"--", "—", "–"}
    band_to_bureau = {"tu": "transunion", "xp": "experian", "eq": "equifax"}
    band_tokens: Dict[str, List[dict]] = {
        "transunion": [],
        "experian": [],
        "equifax": [],
    }
    explicit_dash_for = {"transunion": False, "experian": False, "equifax": False}

    tail_assignments: List[tuple[dict, str]] = []
    for t in tail_tokens:
        band_name = _token_band(t, layout, row_eq_right_x0=row_eq_right_x0)
        tail_assignments.append((t, band_name))
        page_for_log = _int_for_log(t.get("page"))
        line_for_log = _int_for_log(t.get("line"))
        x0_for_log = _float_for_log(t.get("x0"))
        band_label = (band_name or "none").upper()
        text_for_log = t.get("text")
        _triad_band_log(
            "TOK p=%s l=%s x0=%.3f -> %s text=%r",
            page_for_log,
            line_for_log,
            x0_for_log,
            band_label,
            text_for_log if text_for_log is not None else "",
        )
        bureau = band_to_bureau.get(band_name)
        if bureau:
            band_tokens[bureau].append(t)
            try:
                token_text = str(t.get("text", "")).strip()
            except Exception:
                token_text = ""
            if token_text in dash_tokens:
                explicit_dash_for[bureau] = True
    non_label_tail_tokens = [
        t
        for t, _band in tail_assignments
        if not _is_label_token_text(str(t.get("text", "")))
    ]

    if canonical is None and canon_label != "Account #":
        logger.info("TRIAD_GUARD_SKIP reason=unknown_label label=%r", canon_label)
        # Trace label tokens with empty key since it's unknown
        for j, lt in enumerate(label_span):
            _trace_token(lt.get("page"), lt.get("line"), j, lt, "label", "labeled", "")
        # Close any open row to avoid appending future values to the wrong field
        return "CLOSE_OPEN_ROW"

    # Trace label tokens with the resolved key
    for j, lt in enumerate(label_span):
        _trace_token(
            lt.get("page"), lt.get("line"), j, lt, "label", "labeled", canonical or ""
        )

    # 2) Collect values after label suffix, banded by X only
    vals = {"transunion": [], "experian": [], "equifax": []}
    saw_dash_for = {"transunion": False, "experian": False, "equifax": False}
    pre_split_cols: List[str] | None = None
    pre_split_text: str | None = None
    pre_split_mode: str | None = None
    if pre_split_override is not None:
        pre_split_cols = [
            _clean_value(str(part)) if part is not None else ""
            for part in pre_split_override
        ]
        pre_split_text = pre_split_override_text
        pre_split_mode = "override"
    elif (
        SPLIT_SPACE_TRIADS
        and len(non_label_tail_tokens) == 1
        and not any(band_tokens[b] for b in band_tokens)
        and not any(explicit_dash_for.values())
    ):
        candidate = non_label_tail_tokens[0]
        raw = candidate.get("text")
        raw_str = str(raw or "")
        parts = _maybe_split_triad_tail(raw_str)
        if parts:
            pre_split_cols = parts
            pre_split_text = raw_str
            pre_split_mode = "tail_split"
        elif STAGEA_DEBUG and " " in raw_str:
            label_for_log = canonical or canon_label
            _stagea_debug(
                "TRIAD_ROW_FALLBACK account_index=%s label=%s reason=no_delimiter_using_spaces",
                account_index,
                label_for_log,
            )

    if pre_split_cols:
        reason = (
            "single_token_no_geometric_hits"
            if pre_split_mode == "tail_split"
            else pre_split_mode
        )
        logger.info(
            "TRIAD_TAIL_SPACE_SPLIT key=%s text=%r parts=%r reason=%s",
            canonical,
            pre_split_text,
            pre_split_cols,
            reason,
        )
        for j, (t, b) in enumerate(tail_assignments, start=suffix_idx + 1):
            _trace_token(
                t.get("page"),
                t.get("line"),
                j,
                t,
                b,
                "labeled",
                canonical or "",
            )
        for idx, bureau in enumerate(("transunion", "experian", "equifax")):
            part = pre_split_cols[idx] if idx < len(pre_split_cols) else ""
            normalized = clean_value(part)
            vals[bureau] = [normalized]
            if normalized in dash_tokens:
                saw_dash_for[bureau] = True
    else:
        for bureau in ("transunion", "experian", "equifax"):
            if explicit_dash_for[bureau]:
                vals[bureau] = ["--"]
                saw_dash_for[bureau] = True
                continue
            for t in band_tokens[bureau]:
                txt = str(t.get("text", ""))
                vals[bureau].append(txt)
                if txt.strip() in dash_tokens:
                    saw_dash_for[bureau] = True
        for j, (t, b) in enumerate(tail_assignments, start=suffix_idx + 1):
            _trace_token(
                t.get("page"), t.get("line"), j, t, b, "labeled", canonical or ""
            )

    # 2b) TU rescue: sometimes TU values are mis-banded into label due to compression/misalignment.
    # If TU is empty but XP/EQ have values, look for label-band tokens near the TU seam that look like values.
    def _looks_like_tu_value(s: str) -> bool:
        z = (s or "").strip()
        # treat dash placeholders as legitimate values
        if z in {"--", "—", "–"}:
            return True
        # treat dollar amounts as legitimate values
        if z.startswith("$"):
            return True
        # plain integers / with commas / with decimal
        return bool(re.match(r"^[0-9][0-9,]*(?:\.[0-9]+)?$", z))

    if not vals["transunion"] and (vals["experian"] or vals["equifax"]):
        tu_left = float(getattr(layout, "tu_band")[0])
        win_lo = tu_left - 10.0
        win_hi = tu_left + 2.0
        candidates: list[tuple[float, str]] = []
        for j, t in enumerate(tokens[suffix_idx + 1 :], start=suffix_idx + 1):
            if _token_band(t, layout, row_eq_right_x0=row_eq_right_x0) != "label":
                continue
            txt = str(t.get("text", ""))
            z = txt.strip()
            if TRIAD_BAND_BY_X0:
                if z not in dash_tokens:
                    continue
            else:
                if not _looks_like_tu_value(txt):
                    continue
            try:
                midx = _triad_mid_x(t)
            except Exception:
                midx = 0.0
            if win_lo <= midx <= win_hi:
                candidates.append((abs(midx - tu_left), txt, midx))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, picked_text, picked_x = candidates[0]
            if _looks_like_tu_value(picked_text):
                vals["transunion"].append(picked_text)
                if picked_text.strip() in {"--", "—", "–"}:
                    saw_dash_for["transunion"] = True
                    logger.info("TU_DASH_RESCUE text=%r", picked_text)
            logger.info(
                "TRIAD_TU_RESCUE key=%s took=%r from=label near_x=%.1f",
                canonical,
                picked_text,
                picked_x,
            )

    # 2c) Special rule: if TU still empty, pick the first label-band token
    # that looks like a TU value immediately after the label.
    if not vals["transunion"]:
        for j, t in enumerate(tokens[suffix_idx + 1 :], start=suffix_idx + 1):
            if _token_band(
                t, layout, row_eq_right_x0=row_eq_right_x0
            ) != "label":
                continue
            txt = str(t.get("text", ""))
            if not _looks_like_tu_value(txt):
                continue
            vals["transunion"].append(txt)
            if txt.strip() in {"--", "—", "–"}:
                saw_dash_for["transunion"] = True
                logger.info("TU_DASH_RESCUE text=%r", txt)
            try:
                mx = _triad_mid_x(t)
            except Exception:
                mx = 0.0
            logger.info(
                "TRIAD_TU_RESCUE_LABEL key=%s took=%r from=label near_x=%.1f",
                canonical,
                txt,
                mx,
            )
            break

    # 3) Append joined values into fields (no content heuristics)
    def _normalize_value_for_bureau(bureau: str) -> str:
        raw = " ".join(vals[bureau]).strip()
        if not raw:
            return "--" if saw_dash_for[bureau] else ""
        return _clean_value(raw)

    for bureau in triad_order:
        s = _normalize_value_for_bureau(bureau)
        prior = triad_fields[bureau].get(canonical or "", "") if canonical else ""
        triad_fields[bureau][canonical] = (f"{prior} {s}" if prior else s).strip()

    # If we expected values on the next line but actually appended values on this line,
    # clear the expectation flag before returning the row state.
    if expect_values_on_next_line and (
        vals["transunion"] or vals["experian"] or vals["equifax"]
    ):
        expect_values_on_next_line = False

    # Track last bureau that received text on this row, used for wrap affinity
    last_bureau_with_text = None
    for b in ("transunion", "experian", "equifax"):
        if vals[b]:
            last_bureau_with_text = b

    logger.info(
        "TRIAD_ROW_LABELED key=%s TU=%r XP=%r EQ=%r",
        canonical,
        _clean_value(" ".join(vals["transunion"]).strip()),
        _clean_value(" ".join(vals["experian"]).strip()),
        _clean_value(" ".join(vals["equifax"]).strip()),
    )

    if pre_split_mode == "override":
        cols_for_log = [
            _normalize_value_for_bureau("transunion"),
            _normalize_value_for_bureau("experian"),
            _normalize_value_for_bureau("equifax"),
        ]
        label_for_log = canonical or canon_label
        _stagea_debug(
            "TRIAD_ROW_PARSED account_index=%s label=%s cols=%s",
            account_index,
            label_for_log,
            cols_for_log,
        )

    return {
        "triad_row": True,
        "label": _strip_colon_only(visu_label),
        "key": canonical,
        "values": {
            k: _normalize_value_for_bureau(k)
            for k in ("transunion", "experian", "equifax")
        },
        "last_bureau_with_text": last_bureau_with_text,
        "expect_values_on_next_line": expect_values_on_next_line,
    }


def _read_tokens(
    tsv_path: Path,
) -> Tuple[Dict[Tuple[int, int], List[Dict[str, str]]], List[Dict[str, Any]]]:
    """Read tokens from ``tsv_path`` grouped by `(page, line)`.

    Returns a tuple of ``(tokens_by_line, lines)`` where ``tokens_by_line`` maps
    `(page, line)` to the list of token dictionaries, and ``lines`` is an
    ordered list of consolidated line dictionaries containing ``page``,
    ``line`` and joined ``text``.
    """
    tokens_by_line: Dict[Tuple[int, int], List[Dict[str, str]]] = defaultdict(list)
    with tsv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            page_str = row.get("page")
            line_str = row.get("line")
            if not page_str or not line_str:
                continue
            try:
                page = int(float(page_str))
                line = int(float(line_str))
            except Exception:
                # Skip tokens with malformed page/line numbers
                continue
            tokens_by_line[(page, line)].append(row)

    lines: List[Dict[str, Any]] = []
    for page, line in sorted(tokens_by_line.keys()):
        tokens = [tok.get("text", "") for tok in tokens_by_line[(page, line)]]
        if RAW_JOIN_TOKENS_WITH_SPACE:
            text = join_tokens_with_space(tokens)
        else:
            text = "".join(tokens)
        lines.append({"page": page, "line": line, "text": text})
    return tokens_by_line, lines


def find_header_above(
    tokens_by_line: Dict[Tuple[int, int], List[Dict[str, str]]],
    anchor_page: int,
    anchor_line: int,
    anchor_y: float,
) -> List[dict] | None:
    """Return tokens for the nearest header line strictly above the anchor.
    Header must contain exactly {'transunion','experian','equifax'} (any order),
    and nothing else after normalization.
    """

    def _is_pure_triad_header(joined: str) -> bool:
        s = _norm_text(joined)
        parts = s.split()
        names = {"transunion", "experian", "equifax"}
        return (
            len(parts) == 3 and set(parts) == names
        )  # exactly three tokens, no extras

    def _line_header(p: int, ln: int) -> Tuple[List[dict] | None, str | None]:
        toks = tokens_by_line.get((p, ln))
        if not toks:
            return None, None
        # ensure all tokens are above anchor_y
        if any(_mid_y(t) >= anchor_y for t in toks):
            return None, None
        joined = join_tokens_with_space([t.get("text", "") for t in toks])
        if not _is_pure_triad_header(joined):
            return None, None
        return toks, _norm_text(joined)

    # scan upwards on same page
    for ln in range(anchor_line - 1, 0, -1):
        header, norm = _line_header(anchor_page, ln)
        if header:
            ys = sorted(_mid_y(t) for t in header)
            yval = ys[len(ys) // 2] if ys else 0.0
            triad_log(
                "TRIAD_HEADER_ABOVE page=%s line=%s y=%.1f norm=%r",
                anchor_page,
                ln,
                yval,
                norm,
            )
            return header

    # check previous page from last line upwards
    prev_page = anchor_page - 1
    if prev_page >= 1:
        prev_lines = [ln for (pg, ln) in tokens_by_line.keys() if pg == prev_page]
        for ln in sorted(prev_lines, reverse=True):
            toks = tokens_by_line.get((prev_page, ln))
            if not toks:
                continue
            joined = join_tokens_with_space([t.get("text", "") for t in toks])
            if _is_pure_triad_header(joined):
                ys = sorted(_mid_y(t) for t in toks)
                yval = ys[len(ys) // 2] if ys else 0.0
                triad_log(
                    "TRIAD_HEADER_ABOVE page=%s line=%s y=%.1f norm=%r",
                    prev_page,
                    ln,
                    yval,
                    _norm_text(joined),
                )
                return toks

    triad_log(
        "TRIAD_NO_HEADER_ABOVE_ANCHOR page=%s line=%s",
        anchor_page,
        anchor_line,
    )
    return None


def _pick_headline(
    lines: List[Dict[str, Any]], anchor_idx: int, back: int = HEADING_BACK_LINES
) -> Tuple[int, str | None, str]:
    """Return `(start_idx, heading_guess, heading_source)` for an anchor."""

    page = lines[anchor_idx]["page"]

    def _iter_back(start: int):
        for j in range(start, max(anchor_idx - back, -1), -1):
            if lines[j]["page"] != page:
                break
            yield j

    triad_idx: int | None = None
    for j in _iter_back(anchor_idx - 1):
        txt = lines[j]["text"]
        if _is_triad(txt):
            triad_idx = j
            if (
                j - 1 >= 0
                and lines[j - 1]["page"] == page
                and not _is_anchor(lines[j - 1]["text"])
                and _looks_like_headline(lines[j - 1]["text"])
            ):
                return j - 1, lines[j - 1]["text"].strip(), "triad_above"
            break

    for j in _iter_back(anchor_idx - 1):
        txt = lines[j]["text"]
        if _is_anchor(txt) or _is_triad(txt):
            continue
        if _looks_like_headline(txt):
            return j, txt.strip(), "backtrack"

    start_idx = triad_idx if triad_idx is not None else anchor_idx
    return start_idx, None, "anchor_no_heading"


def _find_heading_after_section(
    lines: List[Dict[str, Any]], section_idx: int
) -> Tuple[int, str | None]:
    """Return the index and heading after a section starter line."""

    for j in range(section_idx + 1, min(section_idx + 5, len(lines))):
        text = lines[j]["text"].strip()
        if _is_triad(text) or not _looks_like_headline(text):
            continue
        return j, text

    if section_idx + 1 < len(lines):
        return section_idx + 1, lines[section_idx + 1]["text"].strip()
    return section_idx + 1, None


def _write_account_tsv(
    out_dir: Path,
    account_index: int,
    account_lines: Iterable[Dict[str, Any]],
    tokens_by_line: Dict[Tuple[int, int], List[Dict[str, str]]],
) -> None:
    """Write a debug TSV for a single account."""
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"_debug_account_{account_index}.tsv"
    header = ["page", "line", "y0", "y1", "x0", "x1", "text"]
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(header) + "\n")
        for line in account_lines:
            key = (line["page"], line["line"])
            for tok in tokens_by_line.get(key, []):
                fh.write("\t".join(tok.get(h, "") for h in header) + "\n")


def split_accounts(
    tsv_path: Path, json_out: Path, write_tsv: bool = False
) -> Dict[str, Any]:
    """Core logic for splitting accounts from the full TSV."""
    _triad_x0_fallback_logged.clear()
    tokens_by_line, lines = _read_tokens(tsv_path)
    global trace_dir
    if TRACE_ON:
        trace_dir = Path(
            os.getenv("TRACE_DIR") or (tsv_path.parent / "per_account_tsv")
        )
        trace_dir.mkdir(parents=True, exist_ok=True)
        logger.info("TRACE_DIR=%s", trace_dir.resolve())

    stop_marker_seen = False
    for i, line in enumerate(lines):
        if _norm_simple(line["text"]) == STOP_MARKER_NORM:
            stop_marker_seen = True
            lines = lines[:i]
            break

    anchors = [i for i, line in enumerate(lines) if is_account_anchor(line["text"])]

    account_starts: List[int] = []
    headings: List[str | None] = []
    heading_sources: List[str] = []
    for anchor in anchors:
        start_idx, heading, source = _pick_headline(lines, anchor)
        account_starts.append(start_idx)
        headings.append(heading)
        heading_sources.append(source)

    section_starts = [
        i
        for i, line in enumerate(lines)
        if _norm_simple(line["text"]) in SECTION_STARTERS
    ]
    section_prefix_flags = [False] * len(account_starts)
    sections: List[str | None] = [None] * len(account_starts)

    for s_idx in section_starts:
        next_idx = bisect_right(account_starts, s_idx)
        if next_idx >= len(account_starts):
            continue
        section_prefix_flags[next_idx] = True
        heading_idx, heading = _find_heading_after_section(lines, s_idx)
        account_starts[next_idx] = heading_idx
        headings[next_idx] = heading
        heading_sources[next_idx] = "section+heading"
        starter_norm = _norm_simple(lines[s_idx]["text"])
        sections[next_idx] = _SECTION_NAME.get(starter_norm)

    accounts: List[Dict[str, Any]] = []
    current_section: str | None = None
    section_ptr = 0
    carry_over: List[Dict[str, Any]] = []
    for idx, start_idx in enumerate(account_starts):
        if TRACE_ON:
            _trace_open(f"_trace_account_{idx + 1}.csv")
        if sections[idx] is not None:
            current_section = sections[idx]
        sections[idx] = current_section
        next_start = (
            account_starts[idx + 1] if idx + 1 < len(account_starts) else len(lines)
        )
        while (
            section_ptr < len(section_starts)
            and section_starts[section_ptr] < start_idx
        ):
            section_ptr += 1
        cut_end = next_start
        trailing_pruned = False
        if (
            section_ptr < len(section_starts)
            and start_idx <= section_starts[section_ptr] < next_start
        ):
            cut_end = section_starts[section_ptr]
            trailing_pruned = True
        account_lines = carry_over + lines[start_idx:cut_end]
        carry_over = []
        noise_lines_skipped = 0
        filtered_lines: List[Dict[str, Any]] = []
        for line in account_lines:
            text = line["text"].strip()
            if NOISE_URL_RE.match(text) or NOISE_BANNER_RE.match(text):
                noise_lines_skipped += 1
                continue
            filtered_lines.append(line)
        account_lines = filtered_lines
        history_out: Dict[str, Any] = {}

        def _is_structural_marker(txt: str) -> bool:
            n = _norm_simple(txt)
            return n in SECTION_STARTERS or _is_triad(txt) or _is_anchor(txt)

        while account_lines and _is_structural_marker(account_lines[-1]["text"]):
            carry_over.insert(0, account_lines.pop())
            trailing_pruned = True
        if not account_lines:
            continue
        triad_rows: List[Dict[str, Any]] = []
        triad_maps: Dict[str, Dict[str, str]] = {
            "transunion": {},
            "experian": {},
            "equifax": {},
        }
        pre_split_line_values: Dict[Tuple[int, int], Tuple[Tuple[str, ...], str]] = {}
        in_h2y: bool = False
        in_h7y: bool = False
        current_bureau: Optional[str] = None
        h7y_title_seen: bool = False
        h7y_slabs: Optional[Dict[str, Tuple[float, float]]] = None
        last_key = {"tu": None, "xp": None, "eq": None}
        acc_two_year = {"tu": [], "xp": [], "eq": []}
        acc_seven_year = {
            "tu": {"late30": 0, "late60": 0, "late90": 0},
            "xp": {"late30": 0, "late60": 0, "late90": 0},
            "eq": {"late30": 0, "late60": 0, "late90": 0},
        }
        if RAW_TRIAD_FROM_X:
            open_row: Dict[str, Any] | None = None
            triad_active: bool = False
            current_layout: TriadLayout | None = None
            current_layout_page: int | None = None

            def reset() -> None:
                nonlocal triad_active, current_layout, current_layout_page, open_row
                triad_active = False
                current_layout = None
                current_layout_page = None
                open_row = None

            for line_idx, line in enumerate(account_lines):
                key = (line["page"], line["line"])
                toks = tokens_by_line.get(key, [])
                texts = [t.get("text", "") for t in toks]
                joined_line_text = join_tokens_with_space(texts)
                bare = _bare_bureau_norm(joined_line_text)
                s = _norm_text(joined_line_text)
                n_simple = _norm_simple(joined_line_text)

                line_text_norm = _norm(joined_line_text)

                # --- 7Y gate: require the title first; only then accept the bureau header line ---
                if not in_h7y:
                    if H7Y_TITLE_PAT.search(line_text_norm):
                        h7y_title_seen = True
                    elif h7y_title_seen and all(b.lower() in s for b in H7Y_BUREAUS):
                        mids: Dict[str, float] = {}
                        idxs: Dict[str, int] = {}
                        for idx_t, t in enumerate(toks):
                            txt_norm = _norm(str(t.get("text", ""))).casefold()
                            if txt_norm.startswith("transunion"):
                                mids["tu"] = _triad_mid_x(t)
                                idxs["tu"] = idx_t
                                _history_trace(
                                    t.get("page"),
                                    t.get("line"),
                                    phase="history7y",
                                    text=str(t.get("text", "")),
                                    kind="slab:tu",
                                    x0=t.get("x0"),
                                    x1=t.get("x1"),
                                )
                            elif txt_norm.startswith("experian"):
                                mids["xp"] = _triad_mid_x(t)
                                idxs["xp"] = idx_t
                                _history_trace(
                                    t.get("page"),
                                    t.get("line"),
                                    phase="history7y",
                                    text=str(t.get("text", "")),
                                    kind="slab:xp",
                                    x0=t.get("x0"),
                                    x1=t.get("x1"),
                                )
                            elif txt_norm.startswith("equifax"):
                                mids["eq"] = _triad_mid_x(t)
                                idxs["eq"] = idx_t
                                _history_trace(
                                    t.get("page"),
                                    t.get("line"),
                                    phase="history7y",
                                    text=str(t.get("text", "")),
                                    kind="slab:eq",
                                    x0=t.get("x0"),
                                    x1=t.get("x1"),
                                )
                        if len(mids) == 3:
                            h7y_slabs = {
                                "tu": (0.0, mids["xp"]),
                                "xp": (mids["xp"], mids["eq"]),
                                "eq": (mids["eq"], float("inf")),
                            }
                            in_h7y = True
                            last_key = {"tu": None, "xp": None, "eq": None}
                            logger.info(
                                "H7Y_SLABS tu=[%.1f,%.1f) xp=[%.1f,%.1f) eq=[%.1f,inf)",
                                h7y_slabs["tu"][0],
                                h7y_slabs["tu"][1],
                                h7y_slabs["xp"][0],
                                h7y_slabs["xp"][1],
                                h7y_slabs["eq"][0],
                            )
                            h7y_title_seen = False
                            start_idx = max(idxs.values()) + 1
                            for t in toks[start_idx:]:
                                _process_h7y_token(
                                    t,
                                    h7y_slabs,
                                    acc_seven_year,
                                    last_key,
                                )
                            continue
                        else:
                            # header not clean → do not enter in_h7y
                            h7y_title_seen = False
                            h7y_slabs = None

                # --- 2Y enter condition (full-line regex over normalized text) ---
                if not in_h2y and H2Y_PAT.search(line_text_norm):
                    in_h2y = True
                    current_bureau = None
                    logger.info("H2Y_START page=%s line=%s", line["page"], line["line"])

                # --- line-level stop checks for active history blocks ---
                if in_h2y:
                    stop = (
                        _is_triad(joined_line_text)
                        or is_account_anchor(joined_line_text)
                        or n_simple in SECTION_STARTERS
                        or line_text_norm.lower().startswith(
                            "days late - 7 year history"
                        )
                    )
                    if stop:
                        logger.info(
                            "H2Y_END page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        in_h2y = False
                        current_bureau = None
                        _flush_history(history_out, acc_two_year, acc_seven_year)

                if in_h7y:
                    stop = (
                        _is_triad(joined_line_text)
                        or is_account_anchor(joined_line_text)
                        or bare in BARE_BUREAUS
                        or n_simple in SECTION_STARTERS
                        or H2Y_PAT.search(line_text_norm)
                    )
                    if stop:
                        logger.info(
                            "H7Y_SUMMARY TU=30:%d 60:%d 90:%d XP=30:%d 60:%d 90:%d EQ=30:%d 60:%d 90:%d",
                            acc_seven_year["tu"]["late30"],
                            acc_seven_year["tu"]["late60"],
                            acc_seven_year["tu"]["late90"],
                            acc_seven_year["xp"]["late30"],
                            acc_seven_year["xp"]["late60"],
                            acc_seven_year["xp"]["late90"],
                            acc_seven_year["eq"]["late30"],
                            acc_seven_year["eq"]["late60"],
                            acc_seven_year["eq"]["late90"],
                        )
                        _history_trace(
                            line["page"],
                            line["line"],
                            phase="history7y",
                            kind="tu_summary",
                            text=f"30:{acc_seven_year['tu']['late30']} 60:{acc_seven_year['tu']['late60']} 90:{acc_seven_year['tu']['late90']}",
                        )
                        _history_trace(
                            line["page"],
                            line["line"],
                            phase="history7y",
                            kind="xp_summary",
                            text=f"30:{acc_seven_year['xp']['late30']} 60:{acc_seven_year['xp']['late60']} 90:{acc_seven_year['xp']['late90']}",
                        )
                        _history_trace(
                            line["page"],
                            line["line"],
                            phase="history7y",
                            kind="eq_summary",
                            text=f"30:{acc_seven_year['eq']['late30']} 60:{acc_seven_year['eq']['late60']} 90:{acc_seven_year['eq']['late90']}",
                        )
                        in_h7y = False
                        h7y_slabs = None
                        last_key = {"tu": None, "xp": None, "eq": None}
                        _flush_history(history_out, acc_two_year, acc_seven_year)

                # --- per-token history processing (observer, no continue) ---
                for t in toks:
                    txt_raw = str(t.get("text", ""))
                    txt = _norm(txt_raw)

                    if in_h2y:
                        b = _bureau_key(txt)
                        if b and txt.casefold() in {
                            "transunion",
                            "experian",
                            "equifax",
                        }:
                            current_bureau = b
                            logger.info("H2Y_SET_BUREAU bureau=%s", b.upper())
                        elif current_bureau and H2Y_STATUS_RE.match(txt.casefold()):
                            acc_two_year[current_bureau].append(txt_raw)
                            try:
                                x0 = float(t.get("x0", 0.0))
                            except Exception:
                                x0 = 0.0
                            logger.info(
                                "H2Y_TOKEN bureau=%s text=%r x0=%.1f",
                                current_bureau.upper(),
                                txt_raw,
                                x0,
                            )
                            _trace_write(
                                phase="history2y",
                                bureau=current_bureau,
                                page=line["page"],
                                line=line["line"],
                                text=txt_raw,
                                x0=t.get("x0"),
                                x1=t.get("x1"),
                                mid_x=_triad_mid_x(t),
                            )

                if in_h7y and h7y_slabs:
                    for t in toks:
                        _process_h7y_token(t, h7y_slabs, acc_seven_year, last_key)

                if bare in BARE_BUREAUS:
                    triad_log(
                        "TRIAD_STOP reason=bare_bureau_header page=%s line=%s",
                        line["page"],
                        line["line"],
                    )
                    reset()
                    continue
                if (
                    triad_active
                    and current_layout
                    and current_layout_page is not None
                    and line["page"] != current_layout_page
                ):
                    prev_page = current_layout_page
                    triad_log(
                        "TRIAD_CARRY_PAGE from=%s to=%s",
                        prev_page,
                        line["page"],
                    )
                    current_layout_page = line["page"]

                is_heading_line_without_values = False
                if triad_active and current_layout is not None:
                    is_heading_line_without_values = True
                    for t in toks:
                        b = assign_band(t, current_layout)
                        _trace(line["page"], line["line"], t, b, "heading_check")
                        if b in {"tu", "xp", "eq"}:
                            is_heading_line_without_values = False
                            break
                if triad_active:
                    if s.replace("-", " ") in {"two year payment history"}:
                        triad_log(
                            "TRIAD_STOP reason=twoyearpaymenthistory page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        reset()
                        continue
                    if s.startswith("days late - 7 year history"):
                        triad_log(
                            "TRIAD_STOP reason=days_late page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        reset()
                        continue
                    if (
                        s in {"transunion", "experian", "equifax"}
                        and is_heading_line_without_values
                    ):
                        triad_log(
                            "TRIAD_STOP reason=bare_bureau_header page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        reset()
                        continue
                    if is_account_anchor(joined_line_text):
                        triad_log(
                            "TRIAD_RESET_ON_ANCHOR page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        carry_over = account_lines[line_idx:]
                        account_lines = account_lines[:line_idx]
                        reset()
                        trailing_pruned = True
                        break

                layout: TriadLayout | None = None
                if not triad_active and is_account_anchor(joined_line_text):
                    toks_anchor = tokens_by_line.get((line["page"], line["line"]), [])
                    ys = sorted(_mid_y(t) for t in toks_anchor)
                    anchor_y = ys[len(ys) // 2] if ys else 0.0
                    triad_log(
                        "TRIAD_ANCHOR_AT page=%s line=%s y=%.1f",
                        line["page"],
                        line["line"],
                        anchor_y,
                    )
                    header_toks = find_header_above(
                        tokens_by_line,
                        line["page"],
                        line["line"],
                        anchor_y,
                    )
                    if header_toks:
                        layout = bands_from_header_tokens(header_toks)
                        try:
                            tu_header_x0 = float(getattr(layout, "tu_left_x0", layout.tu_band[0]))
                        except Exception:
                            tu_header_x0 = float(layout.tu_band[0])
                        try:
                            xp_header_x0 = float(getattr(layout, "xp_left_x0", layout.xp_band[0]))
                        except Exception:
                            xp_header_x0 = float(layout.xp_band[0])
                        try:
                            eq_header_x0 = float(getattr(layout, "eq_left_x0", layout.eq_band[0]))
                        except Exception:
                            eq_header_x0 = float(layout.eq_band[0])
                        _triad_band_log(
                            "TRIAD_HEADER_X0S tu=%.3f, xp=%.3f, eq=%.3f",
                            tu_header_x0,
                            xp_header_x0,
                            eq_header_x0,
                        )
                        if STAGEA_DEBUG and header_toks:
                            header_token = header_toks[0]
                            header_page = header_token.get("page")
                            header_line = header_token.get("line")
                            try:
                                header_page = int(float(header_page))
                            except Exception:
                                pass
                            try:
                                header_line = int(float(header_line))
                            except Exception:
                                pass
                            _stagea_debug(
                                "TRIAD_HEADER_DETECTED account_index=%s page=%s line=%s",
                                idx + 1,
                                header_page,
                                header_line,
                            )
                        tu_mid, _ = _band_mid_and_eps(layout.tu_band)
                        xp_mid, _ = _band_mid_and_eps(layout.xp_band)
                        eq_mid, _ = _band_mid_and_eps(layout.eq_band)
                        triad_log(
                            "TRIAD_HEADER_XMIDS tu=%.1f xp=%.1f eq=%.1f",
                            tu_mid,
                            xp_mid,
                            eq_mid,
                        )
                        # When banding by x0, derive left cutoffs from the anchor line's first TU/XP/EQ tokens
                        if TRIAD_BAND_BY_X0:
                            # Find first token in each band after the label
                            first_tu = first_xp = first_eq = None
                            # Use midpoint bands to identify which tokens are TU/XP/EQ on the anchor line
                            seen_label = True
                            for ta in toks_anchor:
                                b = assign_band(ta, layout)
                                if b == "label":
                                    continue
                                if b == "tu" and first_tu is None:
                                    first_tu = ta
                                elif b == "xp" and first_xp is None:
                                    first_xp = ta
                                elif b == "eq" and first_eq is None:
                                    first_eq = ta

                            def _x0(tok):
                                try:
                                    return float(tok.get("x0", 0.0)) if tok else 0.0
                                except Exception:
                                    return 0.0

                            layout.tu_left_x0 = _x0(first_tu)
                            layout.xp_left_x0 = _x0(first_xp)
                            layout.eq_left_x0 = _x0(first_eq)
                            layout.label_right_x0 = layout.tu_left_x0
                            logger.info(
                                "TRIAD_LAYOUT_BOUNDS_X0 label=[0, %.1f) tu=[%.1f, %.1f) xp=[%.1f, %.1f) eq=[%.1f, inf)",
                                layout.label_right_x0,
                                layout.tu_left_x0,
                                layout.xp_left_x0,
                                layout.xp_left_x0,
                                layout.eq_left_x0,
                                layout.eq_left_x0,
                            )
                        logger.info(
                            "TRIAD_LAYOUT_BOUNDS label=[0, %.1f) tu=[%.1f, %.1f) xp=[%.1f, %.1f) eq=[%.1f, inf)",
                            layout.label_band[1],
                            layout.tu_band[0],
                            layout.tu_band[1],
                            layout.xp_band[0],
                            layout.xp_band[1],
                            layout.eq_band[0],
                        )
                        is_valid, anchor_counts = _validate_anchor_row(
                            toks_anchor, layout
                        )
                        if not is_valid:
                            if (
                                header_toks
                                and anchor_counts["label"] == 0
                                and anchor_counts["tu"] >= 1
                                and anchor_counts["xp"] >= 1
                                and anchor_counts["eq"] >= 1
                            ):
                                mids: Dict[str, float] = {}
                                for ht in header_toks:
                                    bkey = _bureau_key(_norm(str(ht.get("text", ""))))
                                    if bkey:
                                        mids[bkey] = _triad_mid_x(ht)
                                if len(mids) == 3:
                                    layout.label_band = (0.0, mids["tu"])
                                    layout.tu_band = (mids["tu"], mids["xp"])
                                    layout.xp_band = (mids["xp"], mids["eq"])
                                    layout.eq_band = (mids["eq"], float("inf"))
                                    if TRIAD_BAND_BY_X0:
                                        if not layout.tu_left_x0:
                                            layout.tu_left_x0 = mids["tu"]
                                        if not layout.xp_left_x0:
                                            layout.xp_left_x0 = mids["xp"]
                                        if not layout.eq_left_x0:
                                            layout.eq_left_x0 = mids["eq"]
                                        if not layout.label_right_x0:
                                            layout.label_right_x0 = layout.tu_left_x0
                                page = line["page"]
                                if page not in _triad_x0_fallback_logged:
                                    logger.info("TRIAD_X0_FALLBACK_OK page=%s", page)
                                    _triad_x0_fallback_logged.add(page)
                                else:
                                    logger.info("TRIAD_X0_FALLBACK_OK page=%s", page)
                                triad_active = True
                                current_layout = layout
                                current_layout_page = page
                                continue
                            logger.info(
                                "TRIAD_STOP reason=layout_mismatch_anchor page=%s line=%s",
                                line["page"],
                                line["line"],
                            )
                            reset()
                            continue
                        triad_active = True
                        current_layout = layout
                        current_layout_page = line["page"]
                elif triad_active and current_layout:
                    layout = current_layout
                    triad_log("TRIAD_CARRY reuse")
                band_tokens: Dict[str, List[dict]] = {
                    "label": [],
                    "tu": [],
                    "xp": [],
                    "eq": [],
                }
                label_token = None
                moved_from_label_on_continuation = False
                pre_split_target: dict | None = None
                if layout:
                    banded_tokens: List[Tuple[dict, str]] = []
                    for t in toks:
                        band = assign_band(t, layout)
                        banded_tokens.append((t, band))
                    bureau_tokens_for_split = [
                        t for t, band in banded_tokens if band in {"tu", "xp", "eq"}
                    ]
                    pre_split_target = (
                        bureau_tokens_for_split[0]
                        if len(bureau_tokens_for_split) == 1
                        else None
                    )
                else:
                    banded_tokens = [(t, "none") for t in toks]

                pre_split_consumed = False
                if layout:
                    for t, band in banded_tokens:
                        _trace(line["page"], line["line"], t, band, "band")
                        pre_split_payload: List[Tuple[str, dict, str]] | None = None
                        if (
                            not pre_split_consumed
                            and pre_split_target is not None
                            and t is pre_split_target
                            and band in {"tu", "xp", "eq"}
                        ):
                            pre_split_payload = _pre_split_bureau_token(t, layout)
                            pre_split_consumed = True
                        if pre_split_payload:
                            parts_for_line: List[str] = []
                            for band_key, synthetic_token, part in pre_split_payload:
                                band_tokens[band_key].append(synthetic_token)
                                parts_for_line.append(part)
                            if key not in pre_split_line_values:
                                pre_split_line_values[key] = (
                                    tuple(parts_for_line),
                                    str(t.get("text", "")),
                                )
                            continue
                        if band in band_tokens:
                            band_tokens[band].append(t)
                        if label_token is None and _has_label_suffix(t.get("text", "")):
                            label_token = t
                    # Detect label-only line (no suffix) before any wrap moves
                    pre_label_tokens = (
                        list(band_tokens["label"]) if band_tokens.get("label") else []
                    )
                    pre_label_txt = join_tokens_with_space(
                        [t.get("text", "") for t in pre_label_tokens]
                    ).strip()
                    if TRIAD_BAND_BY_X0 and label_token is None and pre_label_txt:
                        visu_label = pre_label_txt
                        canon_label = normalize_label_text(visu_label)
                        canonical = LABEL_MAP.get(canon_label)
                        if canonical is not None:
                            for j, lt in enumerate(pre_label_tokens):
                                _trace_token(
                                    line["page"],
                                    line["line"],
                                    j,
                                    lt,
                                    "label",
                                    "labeled",
                                    canonical,
                                )
                            logger.info(
                                "TRIAD_LABEL_LINEBREAK key=%s -> expecting values on next line",
                                canonical,
                            )
                            row_state = {
                                "triad_row": True,
                                "label": _strip_colon_only(visu_label),
                                "key": canonical,
                                "values": {
                                    "transunion": "",
                                    "experian": "",
                                    "equifax": "",
                                },
                                "last_bureau_with_text": None,
                                "expect_values_on_next_line": True,
                            }
                            triad_rows.append(row_state)
                            open_row = row_state
                            continue
                    # Continuations: in x0 mode with no label on this line, do not keep tokens in 'label' band.
                    if (
                        TRIAD_BAND_BY_X0
                        and label_token is None
                        and band_tokens["label"]
                    ):

                        def _nearest_band_from_x0(x0v: float, lay: TriadLayout) -> str:
                            anchors = [
                                (lay.tu_left_x0 or 0.0, "tu"),
                                (lay.xp_left_x0 or 0.0, "xp"),
                                (lay.eq_left_x0 or 0.0, "eq"),
                            ]
                            anchors = [(ax, name) for ax, name in anchors if ax > 0.0]
                            if not anchors:
                                return "tu"
                            return min(
                                ((abs(x0v - ax), name) for ax, name in anchors),
                                key=lambda z: z[0],
                            )[1]

                        moved = {"tu": [], "xp": [], "eq": []}
                        moved_map: Dict[int, str] = {}
                        for lt in list(band_tokens["label"]):
                            try:
                                x0v = float(lt.get("x0", 0.0))
                            except Exception:
                                x0v = 0.0
                            last_bureau = (
                                open_row.get("last_bureau_with_text")
                                if open_row
                                else None
                            )
                            # Choose nearest band, but cap by TRIAD_CONT_NEAREST_MAXDX tolerance.
                            anchors = [
                                (layout.tu_left_x0 or 0.0, "tu"),
                                (layout.xp_left_x0 or 0.0, "xp"),
                                (layout.eq_left_x0 or 0.0, "eq"),
                            ]
                            anchors = [(ax, name) for ax, name in anchors if ax > 0.0]
                            if anchors:
                                dmin, nmin = min(
                                    ((abs(x0v - ax), name) for ax, name in anchors),
                                    key=lambda z: z[0],
                                )
                            else:
                                dmin, nmin = (0.0, "tu")
                            if dmin <= TRIAD_CONT_NEAREST_MAXDX:
                                nb = nmin
                                cause = "nearest"
                            elif last_bureau in {"transunion", "experian", "equifax"}:
                                nb = {
                                    "transunion": "tu",
                                    "experian": "xp",
                                    "equifax": "eq",
                                }[last_bureau]
                                cause = "carry_forward"
                            else:
                                nb = nmin
                                cause = "nearest"
                            moved[nb].append(lt)
                            moved_map[id(lt)] = nb
                            logger.info(
                                "TRIAD_WRAP_AFFINITY key=%s token=%r -> %s cause=%s",
                                (open_row.get("key") if open_row else None),
                                lt.get("text", ""),
                                nb,
                                cause,
                            )
                            _trace_token(
                                line["page"],
                                line["line"],
                                0,
                                lt,
                                nb,
                                "cont",
                                open_row.get("key") if open_row else "",
                                reassigned_from="label",
                                wrap_affinity=cause,
                            )
                        for k in ("tu", "xp", "eq"):
                            if moved[k]:
                                band_tokens[k].extend(moved[k])
                                moved_from_label_on_continuation = True
                        band_tokens["label"] = []
                        # Summary log for continuation wrap reassignment
                        triad_log(
                            "TRIAD_CONT_WRAP page=%s line=%s tu=%d xp=%d eq=%d",
                            line["page"],
                            line["line"],
                            len(moved["tu"]),
                            len(moved["xp"]),
                            len(moved["eq"]),
                        )
                    if (
                        triad_active
                        and ":" in joined_line_text
                        and not band_tokens["label"]
                    ):
                        triad_log(
                            "TRIAD_STOP reason=layout_mismatch page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        triad_active = False
                        current_layout = None
                        current_layout_page = None
                        open_row = None
                        continue
                    if (
                        triad_active
                        and open_row is None
                        and label_token
                        and not in_label_band(label_token, layout)
                    ):
                        triad_log(
                            "TRIAD_STOP reason=layout_mismatch page=%s line=%s",
                            line["page"],
                            line["line"],
                        )
                        reset()
                        continue

                # Drop far-right outliers per band to avoid swallowing trailing noise tokens
                def _filter_band_tokens(
                    btks: List[dict], left_edge: float, window: float = 120.0
                ) -> List[dict]:
                    out: List[dict] = []
                    for _t in btks:
                        try:
                            mx = _triad_mid_x(_t)
                        except Exception:
                            mx = left_edge
                        if mx <= left_edge + window:
                            out.append(_t)
                    return out

                if layout:
                    band_tokens["tu"] = _filter_band_tokens(
                        band_tokens["tu"], layout.tu_band[0]
                    )
                    band_tokens["xp"] = _filter_band_tokens(
                        band_tokens["xp"], layout.xp_band[0]
                    )
                    band_tokens["eq"] = _filter_band_tokens(
                        band_tokens["eq"], layout.eq_band[0]
                    )

                label_txt = join_tokens_with_space(
                    [t.get("text", "") for t in band_tokens["label"]]
                ).strip()
                tu_val = _clean_value(
                    join_tokens_with_space(
                        [t.get("text", "") for t in band_tokens["tu"]]
                    ).strip()
                )
                xp_val = _clean_value(
                    join_tokens_with_space(
                        [t.get("text", "") for t in band_tokens["xp"]]
                    ).strip()
                )
                eq_val = _clean_value(
                    join_tokens_with_space(
                        [t.get("text", "") for t in band_tokens["eq"]]
                    ).strip()
                )
                has_tu = bool(tu_val)
                has_xp = bool(xp_val)
                has_eq = bool(eq_val)

                if not layout:
                    if open_row:
                        triad_log(
                            "TRIAD_GUARD_SKIP page=%s line=%s reason=%s",
                            line["page"],
                            line["line"],
                            "no_layout",
                        )
                        open_row = None
                    continue
                if not label_txt and _is_history_grid_line(band_tokens):
                    triad_log(
                        "TRIAD_STOP reason=grid_line page=%s line=%s",
                        line["page"],
                        line["line"],
                    )
                    reset()
                    continue
                # Label-only line without suffix: open a new row keyed by label and expect values on next line
                if TRIAD_BAND_BY_X0 and label_token is None and label_txt:
                    visu_label = label_txt
                    canon_label = normalize_label_text(visu_label)
                    canonical = LABEL_MAP.get(canon_label)
                    if canonical is not None:
                        # Trace label tokens
                        for j, lt in enumerate(band_tokens["label"]):
                            _trace_token(
                                line["page"],
                                line["line"],
                                j,
                                lt,
                                "label",
                                "labeled",
                                canonical,
                            )
                        logger.info(
                            "TRIAD_LABEL_LINEBREAK key=%s -> expecting values on next line",
                            canonical,
                        )
                        row_state = {
                            "triad_row": True,
                            "label": _strip_colon_only(visu_label),
                            "key": canonical,
                            "values": {"transunion": "", "experian": "", "equifax": ""},
                            "last_bureau_with_text": None,
                            "expect_values_on_next_line": True,
                        }
                        triad_rows.append(row_state)
                        open_row = row_state
                        continue
                if label_token and _is_label_token_text(
                    str(label_token.get("text", ""))
                ):
                    override_cols: List[str] | None = None
                    override_text: str | None = None
                    override_entry = pre_split_line_values.get(key)
                    if override_entry:
                        override_cols = list(override_entry[0])
                        override_text = override_entry[1]
                    row_or_state = process_triad_labeled_line(
                        toks,
                        layout,
                        LABEL_MAP,
                        open_row,
                        triad_maps,
                        ["transunion", "experian", "equifax"],
                        account_index=idx + 1,
                        pre_split_override=override_cols,
                        pre_split_override_text=override_text,
                    )
                    if row_or_state is None:
                        reset()
                        continue
                    elif row_or_state == "CLOSE_OPEN_ROW":
                        open_row = None
                        continue
                    else:
                        triad_rows.append(row_or_state)
                        open_row = row_or_state
                        continue
                else:
                    if open_row:
                        if not (triad_active and current_layout):
                            triad_log(
                                "TRIAD_GUARD_SKIP page=%s line=%s reason=%s",
                                line["page"],
                                line["line"],
                                "triad_inactive",
                            )
                            open_row = None
                            continue
                        if not (has_tu or has_xp or has_eq):
                            triad_log(
                                "TRIAD_GUARD_SKIP page=%s line=%s reason=no_banded_tokens",
                                line["page"],
                                line["line"],
                            )
                            open_row = None
                            continue
                        # Note: do not blanket-skip all single-token lines; see refined guard below
                        # Guard: skip a single short token continuation (likely stray)
                        banded_total = (
                            len(band_tokens["tu"])
                            + len(band_tokens["xp"])
                            + len(band_tokens["eq"])
                        )
                        if (
                            banded_total == 1
                            and len(toks) == 1
                            and not moved_from_label_on_continuation
                        ):
                            only = (
                                band_tokens["tu"]
                                or band_tokens["xp"]
                                or band_tokens["eq"]
                            )
                            if only and len(str(only[0].get("text", "")).strip()) <= 3:
                                triad_log(
                                    "TRIAD_GUARD_SKIP page=%s line=%s reason=short_single_token_continuation",
                                    line["page"],
                                    line["line"],
                                )
                                open_row = None
                                continue
                        # Trace continuation tokens assignment per band

                        # Trace continuation tokens assignment per band
                        if current_layout:
                            for ti, tt in enumerate(toks):
                                # In x0 continuation-wrap mode, prefer the reassigned band if present
                                if (
                                    TRIAD_BAND_BY_X0
                                    and label_token is None
                                    and "moved_map" in locals()
                                ):
                                    bb = moved_map.get(
                                        id(tt), assign_band(tt, current_layout)
                                    )
                                else:
                                    bb = assign_band(tt, current_layout)
                                _trace_token(
                                    line["page"],
                                    line["line"],
                                    ti,
                                    tt,
                                    bb,
                                    "cont",
                                    open_row.get("key") if open_row else "",
                                )
                        appended_any = False
                        if has_tu:
                            open_row["values"][
                                "transunion"
                            ] = f"{open_row['values']['transunion']} {tu_val}".strip()
                            if open_row["key"]:
                                triad_maps["transunion"][open_row["key"]] = open_row[
                                    "values"
                                ]["transunion"]
                            open_row["last_bureau_with_text"] = "transunion"
                            appended_any = True
                        if has_xp:
                            open_row["values"][
                                "experian"
                            ] = f"{open_row['values']['experian']} {xp_val}".strip()
                            if open_row["key"]:
                                triad_maps["experian"][open_row["key"]] = open_row[
                                    "values"
                                ]["experian"]
                            open_row["last_bureau_with_text"] = "experian"
                            appended_any = True
                        if has_eq:
                            open_row["values"][
                                "equifax"
                            ] = f"{open_row['values']['equifax']} {eq_val}".strip()
                            if open_row["key"]:
                                triad_maps["equifax"][open_row["key"]] = open_row[
                                    "values"
                                ]["equifax"]
                            open_row["last_bureau_with_text"] = "equifax"
                            appended_any = True
                        # Task 5: once we've appended any values on the continuation line,
                        # clear the expectation flag for future lines.
                        if appended_any and open_row.get("expect_values_on_next_line"):
                            open_row["expect_values_on_next_line"] = False
                        triad_log(
                            "TRIAD_CONT_PARTIAL page=%s line=%s tu=%s xp=%s eq=%s",
                            line["page"],
                            line["line"],
                            has_tu,
                            has_xp,
                            has_eq,
                        )
        if in_h2y:
            logger.info(
                "H2Y_END page=%s line=%s",
                account_lines[-1]["page"],
                account_lines[-1]["line"],
            )
            in_h2y = False
            current_bureau = None
        if in_h7y:
            logger.info(
                "H7Y_SUMMARY TU=30:%d 60:%d 90:%d XP=30:%d 60:%d 90:%d EQ=30:%d 60:%d 90:%d",
                acc_seven_year["tu"]["late30"],
                acc_seven_year["tu"]["late60"],
                acc_seven_year["tu"]["late90"],
                acc_seven_year["xp"]["late30"],
                acc_seven_year["xp"]["late60"],
                acc_seven_year["xp"]["late90"],
                acc_seven_year["eq"]["late30"],
                acc_seven_year["eq"]["late60"],
                acc_seven_year["eq"]["late90"],
            )
            _history_trace(
                account_lines[-1]["page"],
                account_lines[-1]["line"],
                phase="history7y",
                kind="tu_summary",
                text=f"30:{acc_seven_year['tu']['late30']} 60:{acc_seven_year['tu']['late60']} 90:{acc_seven_year['tu']['late90']}",
            )
            _history_trace(
                account_lines[-1]["page"],
                account_lines[-1]["line"],
                phase="history7y",
                kind="xp_summary",
                text=f"30:{acc_seven_year['xp']['late30']} 60:{acc_seven_year['xp']['late60']} 90:{acc_seven_year['xp']['late90']}",
            )
            _history_trace(
                account_lines[-1]["page"],
                account_lines[-1]["line"],
                phase="history7y",
                kind="eq_summary",
                text=f"30:{acc_seven_year['eq']['late30']} 60:{acc_seven_year['eq']['late60']} 90:{acc_seven_year['eq']['late90']}",
            )
            in_h7y = False
            h7y_slabs = None
            last_key = {"tu": None, "xp": None, "eq": None}

        _flush_history(history_out, acc_two_year, acc_seven_year)

        account_info = {
            "account_index": idx + 1,
            "page_start": account_lines[0]["page"],
            "line_start": account_lines[0]["line"],
            "page_end": account_lines[-1]["page"],
            "line_end": account_lines[-1]["line"],
            "heading_guess": headings[idx],
            "heading_source": heading_sources[idx],
            "section": sections[idx],
            "section_prefix_seen": section_prefix_flags[idx],
            "lines": account_lines,
            "trailing_section_marker_pruned": trailing_pruned,
            "noise_lines_skipped": noise_lines_skipped,
        }
        account_info.update(history_out)
        if RAW_TRIAD_FROM_X:
            account_info["triad"] = {
                "enabled": True,
                "order": ["transunion", "experian", "equifax"],
            }
            triad_fields_out: Dict[str, Dict[str, str]] = {}
            for bureau in ("transunion", "experian", "equifax"):
                filled = ensure_all_keys(triad_maps[bureau])
                triad_fields_out[bureau] = {
                    key: _clean_value(str(value)) if value is not None else ""
                    for key, value in filled.items()
                }
            account_info["triad_fields"] = triad_fields_out
            account_info["triad_rows"] = triad_rows
        accounts.append(account_info)
        if write_tsv:
            _write_account_tsv(
                tsv_path.parent / "per_account_tsv",
                idx + 1,
                account_lines,
                tokens_by_line,
            )

    result = {"accounts": accounts, "stop_marker_seen": stop_marker_seen}
    json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    global _trace_fp, _trace_wr
    if _trace_fp:
        _trace_fp.close()
        _trace_fp = None
        _trace_wr = None
    return result


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Split accounts from the full TSV")
    ap.add_argument("--full", default="_debug_full.tsv", help="Input TSV path")
    ap.add_argument(
        "--json_out", default="accounts_from_full.json", help="JSON output path"
    )
    ap.add_argument(
        "--write-tsv",
        action="store_true",
        help="Write per-account TSVs to per_account_tsv/",
    )
    ap.add_argument(
        "--print-summary",
        action="store_true",
        help="Print a summary of accounts per section",
    )
    ap.add_argument("--sid", help="Run SID for registry")
    ap.add_argument("--manifest", help="Path to runs/<SID>/manifest.json")
    ap.add_argument("--general_out", help="Path for general_info_from_full.json")
    args = ap.parse_args(argv)
    json_out_default = ap.get_default("json_out")

    # Args and env introspection for path resolution transparency
    _dbg_path("ARG_JSON_OUT", getattr(args, "json_out", None))
    _dbg_path("ARG_GENERAL_OUT", getattr(args, "general_out", None))
    _dbg_path("ARG_FULL_TSV", getattr(args, "full", None))
    logger.info("ENV TRIAD_TRACE_CSV=%s", os.getenv("TRIAD_TRACE_CSV"))
    logger.info("ENV KEEP_PER_ACCOUNT_TSV=%s", os.getenv("KEEP_PER_ACCOUNT_TSV"))

    def _sid_from(path_str: str | None) -> str | None:
        if not path_str:
            return None
        try:
            name = Path(path_str).resolve().parent.name
        except Exception:
            return None
        return name or None

    json_out_specified = args.json_out != json_out_default
    sid = (
        args.sid
        or _sid_from(args.manifest)
        or (_sid_from(args.json_out) if json_out_specified else None)
        or _sid_from(args.full)
    )
    if not sid:
        raise ValueError(
            "Unable to determine SID; pass --sid or provide recognizable paths"
        )
    logger.info("RUN_SID %s", sid)

    manifest = RunManifest.for_sid(sid)
    manifest_path_arg = Path(args.manifest).resolve() if args.manifest else None
    if manifest_path_arg and manifest_path_arg != manifest.path.resolve():
        raise ValueError(
            f"Manifest path {manifest_path_arg} does not match runs/{sid} manifest"
        )

    traces_dir = manifest.ensure_run_subdir("traces_dir", "traces")
    accounts_dir = (traces_dir / "accounts_table").resolve()
    accounts_dir.mkdir(parents=True, exist_ok=True)
    low_acct = str(accounts_dir).lower()
    assert ("/runs/" in low_acct) or ("\\runs\\" in low_acct), "Stage-A out_dir must live under runs/<SID>"

    # Finalized base locations for outputs
    _dbg_path("RUN_traces_dir", traces_dir)
    _dbg_path("FINAL_accounts_dir", accounts_dir)

    accounts_json = accounts_dir / "accounts_from_full.json"
    general_json = accounts_dir / "general_info_from_full.json"
    debug_tsv = accounts_dir / "_debug_full.tsv"
    per_acc_dir = (accounts_dir / "per_account_tsv").resolve()
    per_acc_dir.mkdir(parents=True, exist_ok=True)

    _dbg_path("DEST_accounts_json", accounts_json)
    _dbg_path("DEST_general_json", general_json)
    _dbg_path("DEST_debug_full_tsv", debug_tsv)
    _dbg_path("DEST_per_account_tsv_dir", per_acc_dir)

    source_tsv = Path(args.full).resolve()
    if source_tsv != debug_tsv:
        _dbg_path("SRC_full_tsv", source_tsv)
        _dbg_path("DEST_debug_full_tsv", debug_tsv)
        debug_tsv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_tsv, debug_tsv)

    tsv_path = debug_tsv
    json_out = accounts_json
    # Log exact parameters to split_accounts
    _dbg_path("CALL_split_accounts.tsv_path", tsv_path)
    _dbg_path("CALL_split_accounts.json_out", json_out)
    result = split_accounts(tsv_path, json_out, write_tsv=args.write_tsv)
    if args.print_summary:
        accounts = result.get("accounts") or []
        total = len(accounts)
        collections = sum(1 for a in accounts if a.get("section") == "collections")
        unknown = sum(1 for a in accounts if a.get("section") == "unknown")
        regular = total - collections - unknown
        bad_last = [
            a["account_index"]
            for a in accounts
            if a.get("lines")
            and _norm_simple(a["lines"][-1]["text"]) in SECTION_STARTERS
        ]
        print(f"Total accounts: {total}")
        print(f"collections: {collections} unknown: {unknown} regular: {regular}")
        if bad_last:
            print(f"Accounts ending with section starter: {bad_last}")
    print(f"Wrote accounts to {json_out}")

    if not general_json.exists():
        general_json.parent.mkdir(parents=True, exist_ok=True)
        from scripts.split_general_info_from_tsv import split_general_info
        _dbg_path("DEST_general_json", general_json)
        split_general_info(tsv_path, general_json)

    # Enforce canonical outputs under runs/<SID>/traces/accounts_table only.
    # Legacy output args are ignored to avoid duplicating artifacts elsewhere.
    if json_out_specified:
        _dbg_path("IGNORED_ARG_JSON_OUT", getattr(args, "json_out", None))
    if getattr(args, "general_out", None):
        _dbg_path("IGNORED_ARG_GENERAL_OUT", getattr(args, "general_out", None))

    accounts_json_path = accounts_json.resolve()
    general_json_path = general_json.resolve()
    debug_full_tsv_path = tsv_path.resolve()
    base_dir = accounts_dir
    # Always register the canonical per-account TSV dir under acct_dir
    per_account_dir = per_acc_dir

    # Log final resolved artifact locations registered to the manifest
    _dbg_path("DEST_accounts_json", accounts_json_path)
    _dbg_path("DEST_general_json", general_json_path)
    _dbg_path("DEST_debug_full_tsv", debug_full_tsv_path)
    _dbg_path("DEST_per_account_tsv_dir", per_account_dir)

    manifest.set_base_dir("traces_accounts_table", base_dir)
    manifest.snapshot_env(
        [
            "RAW_TRIAD_FROM_X",
            "USE_LAYOUT_TEXT",
            "TRIAD_TRACE_CSV",
            "KEEP_PER_ACCOUNT_TSV",
        ]
    )
    manifest.set_artifact("traces.accounts_table", "accounts_json", accounts_json_path)
    manifest.set_artifact("traces.accounts_table", "general_json", general_json_path)
    manifest.set_artifact("traces.accounts_table", "debug_full_tsv", debug_full_tsv_path)
    manifest.set_artifact(
        "traces.accounts_table", "per_account_tsv_dir", per_account_dir
    )

    # Optional compatibility breadcrumb: if legacy path already exists, drop a .manifest only.
    try:
        legacy_dir = Path("traces") / "blocks" / sid / "accounts_table"
        if legacy_dir.exists():
            write_breadcrumb(manifest.path, legacy_dir / ".manifest")
    except Exception:
        pass

    write_breadcrumb(manifest.path, base_dir / ".manifest")

    logger.info("RUN_MANIFEST_READY sid=%s manifest=%s", manifest.sid, manifest.path)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main()
