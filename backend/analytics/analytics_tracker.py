import atexit
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from backend.api import config
from backend.core.logic.utils.pii import redact_pii

# Cache metrics --------------------------------------------------------------

_CACHE_METRICS: Dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}
_OPS = 0
_SNAPSHOT_INTERVAL = 100

# AI usage metrics -----------------------------------------------------------

_AI_METRICS: Dict[str, float] = {
    "tokens_in": 0,
    "tokens_out": 0,
    "cost": 0.0,
    "latency_ms": 0.0,
}

# Generic counters -----------------------------------------------------------

# Metrics are stored as floats to support both counters and timers.
_COUNTERS: Dict[str, float] = {}


def emit_counter(name: str, increment: float = 1) -> None:
    """Increment a named metric for analytics."""

    _COUNTERS[name] = _COUNTERS.get(name, 0) + increment


def set_metric(name: str, value: float) -> None:
    """Set a named metric to an explicit value."""

    _COUNTERS[name] = value


def get_counters() -> Dict[str, float]:
    """Return current generic metrics (for tests)."""

    return _COUNTERS.copy()


def reset_counters() -> None:
    """Reset generic metrics (for tests)."""

    _COUNTERS.clear()


def _write_cache_snapshot() -> None:
    """Persist current cache metrics to ``analytics_data`` and reset counters."""

    global _OPS
    analytics_dir = Path("analytics_data")
    analytics_dir.mkdir(exist_ok=True)

    now = datetime.now()
    filename = analytics_dir / f"cache_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json"

    payload = {"timestamp": now.isoformat(), "cache": _CACHE_METRICS.copy()}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    for k in _CACHE_METRICS:
        _CACHE_METRICS[k] = 0
    _OPS = 0


def _maybe_flush() -> None:
    if _OPS >= _SNAPSHOT_INTERVAL:
        _write_cache_snapshot()


def _log_cache_event(key: str) -> None:
    global _OPS
    _CACHE_METRICS[key] += 1
    _OPS += 1
    _maybe_flush()


def log_cache_hit() -> None:
    """Record a classification cache hit."""

    _log_cache_event("hits")


def log_cache_miss() -> None:
    """Record a classification cache miss."""

    _log_cache_event("misses")


def log_cache_eviction() -> None:
    """Record a classification cache eviction."""

    _log_cache_event("evictions")


def get_cache_stats() -> Dict[str, int]:
    """Return current cache metrics (for tests)."""

    return _CACHE_METRICS.copy()


def reset_cache_counters() -> None:
    """Reset cache metrics (for tests)."""

    global _OPS
    for k in _CACHE_METRICS:
        _CACHE_METRICS[k] = 0
    _OPS = 0


# AI usage helpers -----------------------------------------------------------

def log_ai_request(tokens_in: int, tokens_out: int, cost: float, latency_ms: float) -> None:
    """Record tokens, estimated cost, and latency for an AI call."""

    _AI_METRICS["tokens_in"] += tokens_in
    _AI_METRICS["tokens_out"] += tokens_out
    _AI_METRICS["cost"] += cost
    _AI_METRICS["latency_ms"] += latency_ms


def get_ai_stats() -> Dict[str, float]:
    """Return current AI usage metrics (for tests)."""

    return _AI_METRICS.copy()


def reset_ai_stats() -> None:
    """Reset AI usage metrics (for tests)."""

    _AI_METRICS.update(tokens_in=0, tokens_out=0, cost=0.0, latency_ms=0.0)


def _flush_on_exit() -> None:
    if _OPS:
        _write_cache_snapshot()


atexit.register(_flush_on_exit)


def save_analytics_snapshot(
    client_info: dict,
    report_summary: dict,
    strategist_failures: Optional[Dict[str, int]] = None,
) -> None:
    logging.getLogger(__name__).info(
        "Analytics tracker using OPENAI_BASE_URL=%s", config.get_ai_config().base_url
    )
    analytics_dir = Path("analytics_data")
    analytics_dir.mkdir(exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M")

    filename = analytics_dir / f"{timestamp}.json"

    snapshot = {
        "date": now.strftime("%Y-%m-%d"),
        "goal": client_info.get("goal", "N/A"),
        "dispute_type": (
            "identity_theft" if client_info.get("is_identity_theft") else "standard"
        ),
        "client_name": client_info.get("name", "Unknown"),
        "client_state": client_info.get("state", "unknown"),
        "summary": {
            "num_collections": report_summary.get("num_collections", 0),
            "num_late_payments": report_summary.get("num_late_payments", 0),
            "high_utilization": report_summary.get("high_utilization", False),
            "recent_inquiries": report_summary.get("recent_inquiries", 0),
            "total_inquiries": report_summary.get("total_inquiries", 0),
            "num_negative_accounts": report_summary.get("num_negative_accounts", 0),
            "num_accounts_over_90_util": report_summary.get(
                "num_accounts_over_90_util", 0
            ),
            "account_types_in_problem": report_summary.get(
                "account_types_in_problem", []
            ),
        },
        "strategic_recommendations": report_summary.get(
            "strategic_recommendations", []
        ),
    }

    if strategist_failures:
        snapshot["strategist_failures"] = strategist_failures

    with open(filename, "w", encoding="utf-8") as f:
        f.write(redact_pii(json.dumps(snapshot, indent=2)))

    print(f"[📊] Analytics snapshot saved: {filename}")
