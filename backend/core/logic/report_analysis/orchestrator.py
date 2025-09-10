from pathlib import Path
import logging

from backend.core.settings import AUTO_PURGE_AFTER_EXPORT
from backend.core.logic.report_analysis.trace_cleanup import purge_after_export
from backend.core.logic.report_analysis import block_exporter

logger = logging.getLogger(__name__)


def run_pipeline(session_id: str) -> dict:
    stage_a = block_exporter.export_stage_a(session_id)
    if not stage_a.get("ok"):
        return {"sid": session_id, "ok": False, "where": "stage_a"}

    result = {"sid": session_id, "ok": True, "artifacts": stage_a["artifacts"]}

    if AUTO_PURGE_AFTER_EXPORT:
        try:
            purge_summary = purge_after_export(sid=session_id, project_root=Path("."))
            logger.info("purge_after_export", extra={"sid": session_id, **purge_summary})
            result["purge"] = purge_summary
        except Exception as e:
            logger.warning(
                "purge_after_export_failed",
                extra={"sid": session_id, "error": str(e)},
            )

    return result
