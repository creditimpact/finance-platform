# Core
## Purpose
Domain logic, models, rules, and services coordinating the credit repair workflow.
## Subfolders / Key Files
- logic/ — analysis algorithms and workflow steps
- models/ — data representations for clients and accounts
- services/ — connectors to external systems
- rules/ — YAML rule definitions
- orchestrators.py — coordinates end-to-end processing
- email_sender.py — notification helpers
## Entry Points
- run_credit_repair_process
## Internal Dependencies
- backend.analytics.analytics_tracker
- backend.audit.audit
- backend.api.tasks
## External Dependencies
- OpenAI APIs (TODO)
- email libraries (TODO)
## Notes / Guardrails
- Handles PII; ensure data is stored and transmitted securely.
