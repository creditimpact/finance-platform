# Analytics

## Purpose
Collects analytics utilities and snapshot tracking used by the platform.

## Files
- `analytics/` – analytics helpers (e.g., strategist failure tallying)
- `analytics_tracker.py` – writes analytics snapshots to disk

## Entry Points
No direct CLI; modules are imported by other backend components.

## Dependencies
- Standard library: `json`, `logging`, `pathlib`, `datetime`
- Internal: `config`

## Notes
Moved from project root during backend reorganization.
