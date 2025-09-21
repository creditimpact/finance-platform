"""AI adjudication pack builder for merge V2 flows."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

from backend.core.io.tags import read_tags
from backend.core.logic.report_analysis.account_merge import get_merge_cfg
from backend.core.logic.report_analysis.keys import normalize_issuer

from . import config as merge_config


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are an adjudicator deciding if A & B are the same account.\n"
    "Consider high-precision cues (account number last4/exact, exact balances with tolerances, dates);"
    " also consider lender/brand name strings and normalize variants (e.g., US BK CACS ≈ U S BANK).\n"
    "Use the numeric summary as a strong hint but override if raw context contradicts it.\n"
    "Decisions allowed: merge, same_debt, different.\n"
    "same_debt when details show the same obligation/story (brand, amounts, dates consistent) but identifiers"
    " differ/are missing (e.g., OC vs CA). If amounts/dates align and descriptors agree, prefer same_debt even"
    " without matching account numbers.\n"
    "Do not choose merge unless evidence strongly supports a single tradeline (e.g., matching account numbers or"
    " multiple tight corroborations).\n"
    "Be conservative when critical fields conflict.\n"
    "Return strict JSON only: {\"decision\":\"merge|same_debt|different\",\"reason\":\"short natural language\"}."
)

MAX_CONTEXT_LINE_LENGTH = 240


FIELD_ORDER: Sequence[str] = (
    "Account #",
    "Balance Owed",
    "Last Payment",
    "Past Due Amount",
    "High Balance",
    "Creditor Type",
    "Account Type",
    "Payment Amount",
    "Credit Limit",
    "Last Verified",
    "Date of Last Activity",
    "Date Reported",
    "Date Opened",
    "Closed Date",
)

REMARK_PREFIXES: Sequence[str] = ("Creditor Remarks", "Remarks")

SKIP_KEYWORDS: Sequence[str] = (
    "two-year payment history",
    "two year payment history",
    "days late - 7 year history",
    "days late-7 year history",
)

HEADER_BUREAU_LINE_RE = re.compile(
    r"(transunion|experian|equifax).*(transunion|experian|equifax)", re.IGNORECASE
)
PAGINATION_RE = re.compile(r"^page\s+\d+\s+of\s+\d+", re.IGNORECASE)
ACCOUNT_NUMBER_RE = re.compile(r"Account #\s*(.*)", re.IGNORECASE)

LENDER_DROP_TOKENS = {
    "LLC",
    "INC",
    "CORP",
    "NA",
    "N A",
    "BANKCARD",
    "CARD",
    "CARDS",
    "SERV",
    "SERVICE",
    "SERVICES",
    "SVCS",
    "CACS",
}


@dataclass(frozen=True)
class _PairTag:
    source_idx: int
    kind: str
    payload: Mapping[str, object]


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


def _load_raw_lines(path: Path) -> List[object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError(f"raw_lines payload must be a list: {path}")


def _normalize_lender_display(raw: str) -> str:
    issuer = normalize_issuer(raw or "")
    if not issuer:
        return ""
    tokens: List[str] = []
    for token in issuer.split():
        adjusted = "BANK" if token == "BK" else token
        if adjusted in LENDER_DROP_TOKENS:
            continue
        tokens.append(adjusted)
    if not tokens:
        tokens = issuer.split()
    normalized = " ".join(tokens)
    return normalized.strip()


def _line_matches_label(line: str, label: str) -> bool:
    normalized_line = line.lower()
    normalized_label = label.lower().rstrip(":")
    return normalized_line.startswith(normalized_label)


def _is_skip_line(line: str) -> bool:
    lowered = line.lower()
    if any(keyword in lowered for keyword in SKIP_KEYWORDS):
        return True
    if PAGINATION_RE.match(lowered):
        return True
    if HEADER_BUREAU_LINE_RE.search(line):
        return True
    return False


def _find_header_line(lines: Sequence[str]) -> str | None:
    for line in lines:
        if not line or _is_only_dashes(line):
            continue
        if _is_skip_line(line):
            continue
        if any(_line_matches_label(line, label) for label in FIELD_ORDER):
            continue
        if any(line.lower().startswith(prefix.lower()) for prefix in REMARK_PREFIXES):
            continue
        return line
    return None


def _collect_field_lines(lines: Sequence[str]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for line in lines:
        if not line or _is_only_dashes(line):
            continue
        if _is_skip_line(line):
            continue
        for label in FIELD_ORDER:
            if label in results:
                continue
            if _line_matches_label(line, label):
                results[label] = line
                break
    return results


def _collect_remarks(lines: Sequence[str]) -> List[str]:
    remarks: List[str] = []
    for line in lines:
        if not line or _is_only_dashes(line):
            continue
        for prefix in REMARK_PREFIXES:
            if line.lower().startswith(prefix.lower()):
                remarks.append(line)
                break
    return remarks


def _extract_account_number(lines: Iterable[str]) -> str | None:
    for line in lines:
        match = ACCOUNT_NUMBER_RE.search(line)
        if not match:
            continue
        tail = match.group(1).strip()
        if not tail:
            continue
        parts = [part.strip(" -:") for part in re.split(r"--", tail)]
        for part in parts:
            cleaned = part.strip()
            if cleaned and not _is_only_dashes(cleaned):
                return cleaned
    return None


def _build_account_context(lines: Sequence[str], max_lines: int) -> List[str]:
    limit = max_lines if max_lines and max_lines > 0 else 0
    if limit <= 0:
        return []

    header = _find_header_line(lines)
    context: List[str] = []
    seen: set[str] = set()

    if header:
        context.append(header)
        seen.add(header)
        normalized = _normalize_lender_display(header)
        if normalized:
            normalized_line = f"Lender normalized: {normalized}"
            if normalized_line not in seen:
                context.append(normalized_line)
                seen.add(normalized_line)

    field_lines = _collect_field_lines(lines)
    for label in FIELD_ORDER:
        line = field_lines.get(label)
        if not line:
            continue
        if line in seen:
            continue
        context.append(line)
        seen.add(line)
        if len(context) >= limit:
            return context[:limit]

    remark_lines = _collect_remarks(lines)
    for line in remark_lines:
        if line in seen:
            continue
        context.append(line)
        seen.add(line)
        if len(context) >= limit:
            break

    return context[:limit]


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def _select_primary_tag(entries: Sequence[_PairTag]) -> Mapping[str, object] | None:
    if not entries:
        return None
    best_entry = None
    best_score = None
    for entry in entries:
        payload = entry.payload
        total = _safe_int(payload.get("total"))
        score = (total, 1 if entry.kind == "merge_pair" else 0)
        if best_score is None or score > best_score:
            best_entry = entry
            best_score = score
    return best_entry.payload if best_entry else None


def _build_highlights(tag_payload: Mapping[str, object]) -> Mapping[str, object]:
    aux = tag_payload.get("aux") if isinstance(tag_payload.get("aux"), Mapping) else {}
    parts = tag_payload.get("parts") if isinstance(tag_payload.get("parts"), Mapping) else {}
    conflicts = (
        list(tag_payload.get("conflicts"))
        if isinstance(tag_payload.get("conflicts"), Sequence)
        else []
    )

    return {
        "total": _safe_int(tag_payload.get("total")),
        "strong": bool(tag_payload.get("strong")),
        "mid_sum": _safe_int(tag_payload.get("mid") or tag_payload.get("mid_sum")),
        "parts": dict(parts),
        "matched_fields": dict(aux.get("matched_fields", {})) if isinstance(aux, Mapping) else {},
        "conflicts": conflicts,
        "acctnum_level": str(aux.get("acctnum_level", "none")) if isinstance(aux, Mapping) else "none",
    }


def _tolerance_hint() -> Mapping[str, float | int]:
    cfg = get_merge_cfg()
    tolerances = cfg.tolerances if isinstance(cfg.tolerances, Mapping) else {}

    def _get_float(key: str, default: float) -> float:
        raw = tolerances.get(key)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float(default)

    def _get_int(key: str, default: int) -> int:
        raw = tolerances.get(key)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return int(default)

    return {
        "amount_abs_usd": _get_float("AMOUNT_TOL_ABS", 50.0),
        "amount_ratio": _get_float("AMOUNT_TOL_RATIO", 0.01),
        "last_payment_day_tol": _get_int("LAST_PAYMENT_DAY_TOL", 7),
    }


def _load_account_payload(
    accounts_root: Path,
    account_idx: int,
    cache: MutableMapping[int, Mapping[str, object]],
    max_lines: int,
) -> Mapping[str, object]:
    if account_idx in cache:
        return cache[account_idx]

    raw_path = accounts_root / str(account_idx) / "raw_lines.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_lines.json not found for account {account_idx}")

    raw_lines = _load_raw_lines(raw_path)
    normalized_lines = [_normalize_line(_coerce_text(line)) for line in raw_lines]
    context = _build_account_context(normalized_lines, max_lines)
    account_number = _extract_account_number(normalized_lines)

    payload = {"context": context, "account_number": account_number, "lines": normalized_lines}
    cache[account_idx] = payload
    return payload


def _collect_pair_entries(accounts_root: Path) -> tuple[Dict[tuple[int, int], List[_PairTag]], Dict[int, set[int]]]:
    pair_entries: Dict[tuple[int, int], List[_PairTag]] = {}
    best_partners: Dict[int, set[int]] = {}

    allowed_kinds = {"merge_pair", "merge_best"}

    for entry in sorted(accounts_root.iterdir(), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        try:
            idx = int(entry.name)
        except ValueError:
            continue

        tags_path = entry / "tags.json"
        tags = read_tags(tags_path)
        for tag in tags:
            if not isinstance(tag, Mapping):
                continue
            kind = str(tag.get("kind"))
            if kind not in allowed_kinds:
                continue
            decision = str(tag.get("decision", "")).lower()
            if decision != "ai":
                continue
            partner = tag.get("with")
            try:
                partner_idx = int(partner)
            except (TypeError, ValueError):
                continue
            pair_key = tuple(sorted((idx, partner_idx)))
            pair_entries.setdefault(pair_key, []).append(
                _PairTag(source_idx=idx, kind=kind, payload=tag)
            )
            if kind == "merge_best":
                best_partners.setdefault(idx, set()).add(partner_idx)

    return pair_entries, best_partners


def _should_include_pair(
    pair: tuple[int, int],
    best_partners: Mapping[int, set[int]],
    only_merge_best: bool,
) -> bool:
    if not only_merge_best:
        return True
    a, b = pair
    best_a = best_partners.get(a, set())
    best_b = best_partners.get(b, set())
    return (b in best_a) or (a in best_b)


def build_merge_ai_packs(
    sid: str,
    runs_root: Path | str,
    *,
    only_merge_best: bool = True,
    max_lines_per_side: int = 20,
) -> List[Mapping[str, object]]:
    """Build merge AI packs for ``sid``.

    Parameters
    ----------
    sid:
        The session identifier.
    runs_root:
        Root directory containing the ``runs/<sid>`` layout.
    only_merge_best:
        When ``True`` include only pairs that appear as ``merge_best`` partners.
    max_lines_per_side:
        Maximum number of context lines per account. Values greater than the
        ``AI_PACK_MAX_LINES_PER_SIDE`` environment configuration are clamped to
        that cap.
    """

    sid_str = str(sid)
    runs_root_path = Path(runs_root)
    accounts_root = runs_root_path / sid_str / "cases" / "accounts"

    if not accounts_root.exists():
        raise FileNotFoundError(
            f"cases/accounts directory not found for sid={sid_str!r} under {runs_root_path}"
        )

    env_limit = merge_config.get_ai_pack_max_lines_per_side()
    requested_limit = max_lines_per_side if max_lines_per_side and max_lines_per_side > 0 else env_limit
    context_limit = min(env_limit, requested_limit) if env_limit > 0 else max(requested_limit, 1)

    pair_entries, best_partners = _collect_pair_entries(accounts_root)
    tolerance_hint = _tolerance_hint()

    cache: Dict[int, Mapping[str, object]] = {}
    packs: List[Mapping[str, object]] = []

    for pair in sorted(pair_entries.keys()):
        if not _should_include_pair(pair, best_partners, only_merge_best):
            continue

        entries = pair_entries.get(pair, [])
        primary_tag = _select_primary_tag(entries)
        if primary_tag is None:
            logger.warning("MERGE_V2_PACK_MISSING_TAG sid=%s pair=%s", sid_str, pair)
            continue

        a_idx, b_idx = pair

        try:
            account_a = _load_account_payload(accounts_root, a_idx, cache, context_limit)
            account_b = _load_account_payload(accounts_root, b_idx, cache, context_limit)
        except FileNotFoundError as exc:
            logger.warning(
                "MERGE_V2_PACK_MISSING_LINES sid=%s pair=%s error=%s",
                sid_str,
                pair,
                exc,
            )
            continue

        context_a = list(account_a.get("context", []))[:context_limit]
        context_b = list(account_b.get("context", []))[:context_limit]
        highlights = _build_highlights(primary_tag)

        account_number_a = account_a.get("account_number")
        account_number_b = account_b.get("account_number")
        ids_payload = {
            "account_number_a": account_number_a if account_number_a else None,
            "account_number_b": account_number_b if account_number_b else None,
        }

        summary = dict(highlights)

        user_payload = {
            "sid": sid_str,
            "pair": {"a": a_idx, "b": b_idx},
            "ids": ids_payload,
            "numeric_match_summary": summary,
            "tolerances_hint": dict(tolerance_hint),
            "limits": {"max_lines_per_side": context_limit},
            "context": {"a": context_a, "b": context_b},
            "output_contract": {
                "decision": ["merge", "same_debt", "different"],
                "reason": "short natural language",
            },
        }

        pack = {
            "sid": sid_str,
            "pair": {"a": a_idx, "b": b_idx},
            "ids": ids_payload,
            "highlights": summary,
            "tolerances_hint": dict(tolerance_hint),
            "limits": {"max_lines_per_side": context_limit},
            "context": {"a": context_a, "b": context_b},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
                },
            ],
        }

        packs.append(pack)

    return packs


__all__ = ["build_merge_ai_packs"]

