"""Pipeline lifecycle hooks for cross-stage orchestration."""

from __future__ import annotations


def on_cases_built(sid: str) -> None:
    """Trigger frontend pack generation when cases are materialised."""

    from backend.api.tasks import generate_frontend_packs_task

    generate_frontend_packs_task.apply_async(args=[sid], queue="frontend")
