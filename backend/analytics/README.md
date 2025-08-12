# Analytics

## Purpose
Capture lightweight metrics and snapshots about each credit repair run for internal diagnostics.

## Pipeline position
Invoked after report analysis and strategy generation to persist summary statistics and failure reasons.

## Files
- `__init__.py`: package marker.
- `analytics_tracker.py`: write JSON analytics snapshots to disk.
  - Key function: `save_analytics_snapshot()`.
  - Internal deps: `backend.api.config`.
- `analytics/` – helper subpackage (e.g., `strategist_failures.py`) providing counters; no separate README.

## Entry points
- `analytics_tracker.save_analytics_snapshot`

## Guardrails / constraints
- Intended for internal use only; avoid storing sensitive client data in snapshots.
