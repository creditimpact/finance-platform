# Audit

## Purpose
Provides auditing and trace export utilities for strategy runs.

## Files
- `audit.py` – structured audit logger and helpers
- `trace_exporter.py` – exports trace diagnostics and per-account breakdowns

## Entry Points
No standalone entry point; used by backend services.

## Dependencies
- Standard library: `json`, `datetime`, `pathlib`
- Internal: `models.strategy`

## Notes
Moved from project root during backend reorganization.
