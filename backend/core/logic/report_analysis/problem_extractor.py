from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from backend.settings import PROJECT_ROOT

from .keys import compute_logical_account_key
from .problem_detection import evaluate_account_problem

logger = logging.getLogger(__name__)


def _get_field(account: Mapping[str, Any], *names: str) -> Any:
    """Return the first non-null field matching ``names`` from account or its fields."""
    fields = account.get("fields") if isinstance(account.get("fields"), Mapping) else None
    for name in names:
        if fields and fields.get(name) is not None:
            return fields.get(name)
        if account.get(name) is not None:
            return account.get(name)
    return None


def _make_account_id(account: Mapping[str, Any], idx: int) -> str:
    issuer = _get_field(account, "issuer", "creditor", "name")
    last4 = _get_field(account, "account_last4", "last4")
    account_type = _get_field(account, "account_type", "type")
    opened_date = _get_field(account, "opened_date", "date_opened")
    key = compute_logical_account_key(issuer, last4, account_type, opened_date)
    acc_id = key or f"idx-{idx:03d}"
    return re.sub(r"[^a-z0-9_-]", "_", acc_id.lower())


def _normalize_signal(sig: Any) -> List[str]:
    out: List[str] = []
    if isinstance(sig, Mapping):
        out.extend(str(k).replace(" ", "_").lower() for k in sig.keys())
    else:
        s = str(sig)
        if s.startswith("status_present:"):
            s = s.split(":", 1)[1]
        s = re.sub(r"^status_", "", s)
        out.append(s.replace(" ", "_").lower())
    return out


def detect_problem_accounts(sid: str, root: Path | None = None) -> List[Dict[str, Any]]:
    """Return problematic accounts for ``sid`` based on rule evaluation."""
    base = Path(root or PROJECT_ROOT)
    logger.info("PROBLEM_EXTRACT start sid=%s", sid)
    acc_path = base / "traces" / "blocks" / sid / "accounts_table" / "accounts_from_full.json"
    accounts: List[Mapping[str, Any]] = []
    if acc_path.exists():
        try:
            data = json.loads(acc_path.read_text(encoding="utf-8"))
            if isinstance(data, Mapping):
                accounts = list(data.get("accounts") or [])
            elif isinstance(data, list):
                accounts = list(data)
        except Exception:
            accounts = []
    total = len(accounts)
    results: List[Dict[str, Any]] = []
    for i, account in enumerate(accounts, start=1):
        if not isinstance(account, Mapping):
            continue
        account_id = _make_account_id(account, i)
        fields = account.get("fields") if isinstance(account.get("fields"), Mapping) else account
        decision = evaluate_account_problem(dict(fields))
        problem_reasons = list(decision.get("problem_reasons") or [])
        signals = (decision.get("debug", {}) if isinstance(decision.get("debug"), Mapping) else {}).get("signals") or []
        tags: List[str] = []
        for sig in signals:
            for t in _normalize_signal(sig):
                if t not in tags:
                    tags.append(t)
        confidence: float | None = None
        if "confidence" in decision:
            try:
                confidence = float(decision["confidence"])
            except Exception:
                confidence = None
        if problem_reasons or tags:
            item: Dict[str, Any] = {
                "account_id": account_id,
                "account_index": i,
                "problem_tags": tags,
                "problem_reasons": problem_reasons,
            }
            if confidence is not None:
                item["confidence"] = confidence
            results.append(item)
    logger.info(
        "PROBLEM_EXTRACT done sid=%s total=%s problematic=%s",
        sid,
        total,
        len(results),
    )
    return results
