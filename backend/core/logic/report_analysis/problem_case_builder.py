from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from backend.settings import PROJECT_ROOT
from .keys import compute_logical_account_key

logger = logging.getLogger(__name__)


def _make_account_id(account: Mapping[str, Any], idx: int) -> str:
    """Return a filesystem-friendly account identifier.

    Attempts to compute a stable logical key using identifying fields when
    available.  Falls back to a deterministic surrogate based on the heading
    slug or account index.
    """

    fields = account.get("fields") or {}

    issuer = (
        fields.get("issuer")
        or fields.get("creditor")
        or fields.get("name")
        or account.get("issuer")
        or account.get("creditor")
        or account.get("name")
    )
    last4 = (
        fields.get("account_last4")
        or fields.get("last4")
        or account.get("account_last4")
        or account.get("last4")
    )
    account_type = (
        fields.get("account_type")
        or fields.get("type")
        or account.get("account_type")
        or account.get("type")
    )
    opened_date = (
        fields.get("opened_date")
        or fields.get("date_opened")
        or account.get("opened_date")
        or account.get("date_opened")
    )

    logical_key = compute_logical_account_key(issuer, last4, account_type, opened_date)
    if logical_key:
        return logical_key

    heading = account.get("heading_guess")
    if heading:
        raw = heading
    else:
        raw = f"account_{idx}"

    return re.sub(r"[^A-Za-z0-9._-]", "_", str(raw))


def _derive_problems(
    account: Mapping[str, Any]
) -> tuple[list[str], list[str], float | None]:
    """Return problem tags, reasons, and optional confidence for ``account``.

    This basic implementation flags accounts missing a heading.  Future tasks
    expand on this detection logic.
    """
    tags: list[str] = []
    reasons: list[str] = []
    confidence: float | None = None
    if not account.get("heading_guess"):
        tags.append("missing_heading")
        reasons.append("missing heading")
    return tags, reasons, confidence


def build_problem_cases(session_id: str, root: Path | None = None) -> dict:
    """Detect problematic accounts from ``accounts_from_full.json``.

    Parameters
    ----------
    session_id:
        Identifier used to locate ``traces/blocks/<sid>``.
    root:
        Repository root; defaults to :data:`~backend.settings.PROJECT_ROOT`.
    """

    logger.info("PROBLEM_CASES start sid=%s", session_id)

    base = (root or PROJECT_ROOT) / "traces" / "blocks" / session_id / "accounts_table"
    acc_path = base / "accounts_from_full.json"
    accounts: list[MutableMapping[str, Any]] = []
    if acc_path.exists():
        try:
            data = json.loads(acc_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                accounts = list(data.get("accounts") or [])
            elif isinstance(data, list):
                accounts = list(data)
        except Exception:
            accounts = []

    total = len(accounts)

    out_dir = (root or PROJECT_ROOT) / "cases" / session_id
    accounts_out = out_dir / "accounts"
    accounts_out.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for idx, acc in enumerate(accounts, start=1):
        if not isinstance(acc, Mapping):
            continue
        account_id = _make_account_id(acc, idx)
        tags, reasons, confidence = _derive_problems(acc)
        if not tags and not reasons:
            continue
        case = {
            "sid": session_id,
            "account_id": account_id,
            "source": "accounts_from_full",
            "account": acc,
            "problem_tags": tags,
            "problem_reasons": reasons,
        }
        if confidence is not None:
            case["confidence"] = confidence
        (accounts_out / f"{account_id}.json").write_text(
            json.dumps(case, indent=2), encoding="utf-8"
        )
        summaries.append(
            {
                "account_id": account_id,
                "problem_tags": tags,
                "problem_reasons": reasons,
            }
        )

    index = {
        "sid": session_id,
        "total": total,
        "problematic": len(summaries),
        "problematic_accounts": summaries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    logger.info(
        "PROBLEM_CASES done sid=%s total=%s problematic=%s",
        session_id,
        total,
        len(summaries),
    )

    return {
        "sid": session_id,
        "total": total,
        "problematic": len(summaries),
        "out_dir": str(out_dir),
        "summaries": summaries,
    }
