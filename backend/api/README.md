# API
## Purpose
Hosts the Flask API layer with admin routes, session helpers, Celery tasks, and configuration utilities.
## Subfolders / Key Files
- __init__.py — package marker
- app.py — main API blueprint and routes
- admin.py — admin-only endpoints
- tasks.py — Celery worker tasks
- session_manager.py — JSON-backed session store
- config.py — environment-driven configuration
- telegram_alert.py — console alert for admin logins
## Entry Points
- api_bp.start_process
- api_bp.explanations_endpoint
- api_bp.get_summaries
- admin_bp.login
- extract_problematic_accounts task
## Internal Dependencies
- backend.core.orchestrators
- backend.core.models
- backend.core.logic.explanations_normalizer
## External Dependencies
- Flask
- Flask-CORS
- Celery
## Notes / Guardrails
- Session data should contain only sanitized client input.
- Secrets come from environment variables; never commit credentials.
