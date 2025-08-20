# Analytics

## Purpose
Capture lightweight metrics and snapshots about each credit repair run for internal diagnostics.

## Pipeline position
Invoked after report analysis and strategy generation to persist summary statistics and failure reasons.

## Files
- `__init__.py`: package marker.
- `analytics_tracker.py`: write JSON analytics snapshots to disk.
  - Key function: `save_analytics_snapshot()`.
  - Exposes cache counters: `log_cache_hit`, `log_cache_miss`, `log_cache_eviction`.
  - Writes cache metrics JSON every 100 events or on shutdown.
  - Internal deps: `backend.api.config`.
- `analytics/` – helper subpackage (e.g., `strategist_failures.py`) providing counters; no separate README.

### Monitoring counters

Common counters emitted for dashboards:

- `letters_without_strategy_context` – attempts to generate letters without required strategy data.
- `guardrail_fix_count.{letter_type}` – number of guardrail remediation passes by letter type.
- `policy_override_reason.{reason}` – policy-based overrides grouped by reason.
- `rulebook.tag_selected.{tag}` – counts how often a rulebook action tag is chosen.
- `rulebook.suppressed_rules.{rule_name}` – rules skipped due to precedence or exclusion.

## Entry points
- `analytics_tracker.save_analytics_snapshot`

## Guardrails / constraints
- Intended for internal use only; avoid storing sensitive client data in snapshots.

### Cache metrics example

```
{
  "timestamp": "2024-05-06T12:00:00",
  "cache": {"hits": 10, "misses": 2, "evictions": 1}
}
```
