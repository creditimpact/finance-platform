# Core

## Purpose
The core layer contains the domain logic and orchestration for credit repair processes. It houses business rules, services, and data models that drive automated workflows.

## Subfolders
- `logic/` – analysis algorithms and workflow steps for credit reports and strategy generation.
- `models/` – data models representing clients, accounts, bureaus, and related entities.
- `services/` – interfaces to external systems and supporting service clients.
- `rules/` – YAML rule definitions for disputes, compliance, and neutral phrasing.

## Root files
- `orchestrators.py` – coordinates the end-to-end credit repair process.
- `email_sender.py` – utilities for sending process and notification emails.

## Entry points
- `run_credit_repair_process` – drives the full credit repair workflow.

## Key dependencies
- Internal: `analytics_tracker`, `audit`, `config`, API task layer.
- External: networking, OpenAI, and other third‑party libraries.

## Notes / guardrails
Handles sensitive credit data; ensure PII is managed securely and follow compliance requirements when extending or using these modules.
