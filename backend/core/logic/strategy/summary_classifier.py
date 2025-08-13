import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from backend.core.logic.utils.json_utils import parse_json
from backend.core.services.ai_client import AIClient

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    value: Mapping[str, str]
    timestamp: float
    ttl: float | None


_CACHE: Dict[Tuple[str, str, str], _CacheEntry] = {}


def _summary_hash(summary: Mapping[str, Any]) -> str:
    data = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _cache_get(
    session_id: str, account_id: str, summary: Mapping[str, Any]
) -> Mapping[str, str] | None:
    key = (session_id, account_id, _summary_hash(summary))
    entry = _CACHE.get(key)
    if not entry:
        return None
    if entry.ttl is not None and time.time() - entry.timestamp > entry.ttl:
        _CACHE.pop(key, None)
        return None
    return entry.value


def _cache_set(
    session_id: str,
    account_id: str,
    summary: Mapping[str, Any],
    value: Mapping[str, str],
    ttl: float | None,
) -> None:
    key = (session_id, account_id, _summary_hash(summary))
    _CACHE[key] = _CacheEntry(value=value, timestamp=time.time(), ttl=ttl)


def invalidate_summary_cache(session_id: str, account_id: str | None = None) -> None:
    keys = [
        k
        for k in list(_CACHE)
        if k[0] == session_id and (account_id is None or k[1] == account_id)
    ]
    for k in keys:
        _CACHE.pop(k, None)


_RULE_MAP = {
    "identity_theft": {
        "legal_tag": "FCRA §605B",
        "dispute_approach": "fraud_block",
        "tone": "urgent",
    },
    "not_mine": {
        "legal_tag": "FCRA §609(e)",
        "dispute_approach": "validation",
        "tone": "firm",
    },
    "goodwill": {
        "legal_tag": "FCRA §623(a)(1)",
        "dispute_approach": "goodwill_adjustment",
        "tone": "conciliatory",
    },
    "inaccurate_reporting": {
        "legal_tag": "FCRA §611",
        "dispute_approach": "reinvestigation",
        "tone": "professional",
    },
}

_STATE_HOOKS = {
    "CA": "California Consumer Credit Reporting Agencies Act",
    "NY": "New York FCRA Article 25",
}


def _heuristic_category(summary: Mapping[str, Any]) -> str:
    text_bits = [
        summary.get("dispute_type", ""),
        summary.get("facts_summary", ""),
    ] + summary.get("claimed_errors", [])
    text = " ".join([t.lower() for t in text_bits if isinstance(t, str)])
    if "identity" in text or "stolen" in text:
        return "identity_theft"
    if "not mine" in text:
        return "not_mine"
    if "goodwill" in text:
        return "goodwill"
    return "inaccurate_reporting"


def classify_client_summary(
    summary: Mapping[str, Any],
    ai_client: AIClient,
    state: str | None = None,
    *,
    session_id: str | None = None,
    account_id: str | None = None,
    ttl: float | None = 24 * 3600,
) -> Mapping[str, str]:
    """Classify a structured summary into a dispute category and legal strategy.

    When ``session_id`` and ``account_id`` are provided the result is cached
    using those identifiers and a hash of ``summary``.
    """

    if session_id and account_id:
        cached = _cache_get(session_id, account_id, summary)
        if cached:
            return cached

    category = None
    prompt = (
        "Classify the following structured credit dispute summary into one of "
        "the categories: not_mine, inaccurate_reporting, identity_theft, goodwill. "
        "Return only JSON with a 'category' field. Summary: "
        f"{summary}"
    )
    try:
        resp = ai_client.response_json(
            prompt=prompt,
            response_format={"type": "json_object"},
        )
        content = resp.output[0].content[0].text
        data, _ = parse_json(content)
        data = data or {}
        category = data.get("category")
    except Exception:
        category = None
    if not category:
        category = _heuristic_category(summary)

    mapping = _RULE_MAP.get(category, _RULE_MAP["inaccurate_reporting"]).copy()
    result = {"category": category, **mapping}
    if state and state in _STATE_HOOKS:
        result["state_hook"] = _STATE_HOOKS[state]
    logger.info("Summary classification: %s -> %s", summary.get("account_id"), result)
    if session_id and account_id:
        _cache_set(session_id, account_id, summary, result, ttl)
    return result


__all__ = ["classify_client_summary", "invalidate_summary_cache"]
