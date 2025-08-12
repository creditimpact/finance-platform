# API Layer

## Purpose
Provides Flask app and admin routes, Celery tasks, session management, configuration helpers, and alert utilities.

## Files
- `__init__.py` – package marker.
- `app.py` – main API blueprint with public routes.
- `admin.py` – admin-only Flask blueprint and helpers.
- `tasks.py` – Celery worker tasks for report processing.
- `session_manager.py` – simple JSON-backed session storage.
- `config.py` – environment-driven application configuration.
- `telegram_alert.py` – console alert for admin logins.

## Entry points
- Flask routes: `api_bp.index`, `api_bp.start_process`, `api_bp.explanations_endpoint`, `api_bp.get_summaries`.
- Admin routes: `admin_bp.login`, `admin_bp.logout`, `admin_bp.index`, `admin_bp.download_client`, `admin_bp.download_analytics`.
- Celery tasks: `extract_problematic_accounts`, `process_report`.
- Session helpers: `set_session`, `get_session`, `update_session`, `update_intake`, `get_intake`.
- Config accessors: `get_app_config`, `get_ai_config`.
- Alert: `send_admin_login_alert`.

## Key dependencies
- Internal: `orchestrators`, `models`, `logic.explanations_normalizer`, `services.ai_client`.
- External: `Flask`, `Flask-CORS`, `Celery`, `uuid`, `dataclasses`.

## Notes/guardrails
- Intake storage must only hold raw client explanations; downstream components should use sanitized summaries.
- Environment variables (e.g., `OPENAI_API_KEY`, `ADMIN_PASSWORD`) are required; avoid committing secrets.
- Admin login alerts are logged locally to prevent unnecessary network calls.

## Likely to update later
- Import paths throughout the repo referencing `app`, `admin`, `tasks`, `session_manager`, `config`, or `telegram_alert` will need to change to `backend.api.*` (e.g., `orchestrators.py`, `logic/*`, tests).
