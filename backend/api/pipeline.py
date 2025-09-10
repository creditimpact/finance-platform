from backend.api.tasks import stage_a_task, cleanup_trace_task, extract_problematic_accounts


def run_full_pipeline(sid: str):
    """Run Stage-A export, cleanup traces, and extract accounts for ``sid``.

    Each task receives ``sid`` explicitly via ``.si`` to avoid relying on the
    previous task's return value.
    """
    return (
        stage_a_task.si(sid)
        | cleanup_trace_task.si(sid)
        | extract_problematic_accounts.si(sid)
    ).apply_async()
