"""Pipeline lifecycle hooks for cross-stage orchestration."""

from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)


def on_cases_built(sid: str) -> None:
    """Trigger frontend pack generation when cases are materialised."""

    if os.getenv("FRONTEND_TRIGGER_AFTER_CASES", "1") != "1":
        logger.info("REVIEW_TRIGGER: skip_enqueue sid=%s reason=env_disabled", sid)
        return

    from backend.api.tasks import generate_frontend_packs_task

    generate_frontend_packs_task.delay(sid)
