from celery import chain

from backend.api.tasks import (
    cleanup_trace_task,
    extract_problematic_accounts,
    stage_a_task,
)


def run_full_pipeline(sid: str):
    """Run Stage-A export, cleanup traces, and extract accounts for ``sid``.

    Each task receives ``sid`` explicitly via ``.si`` to avoid relying on the
    previous task's return value.
    """
    return chain(
        stage_a_task.si(sid),
        cleanup_trace_task.si(sid),
        extract_problematic_accounts.si(sid),
    ).apply_async()
