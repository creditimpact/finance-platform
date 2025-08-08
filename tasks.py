# ruff: noqa: E402
import os
import sys
import uuid
import logging

# Ensure the project root is always on sys.path so local modules can be
# imported even when the worker is launched from outside the repository
# directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from celery import Celery
from orchestrators import (
    run_credit_repair_process,
    extract_problematic_accounts_from_report,
)
from config import get_app_config

_app_config = get_app_config()
app = Celery(
    "tasks",
    broker=_app_config.celery_broker_url,
    backend=_app_config.celery_broker_url,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Celery worker starting with OPENAI_BASE_URL=%s", _app_config.ai.base_url)
logger.info("Celery worker OPENAI_API_KEY present=%s", bool(_app_config.ai.api_key))

# Verify that session_manager is importable at startup. This helps catch
# cases where the worker is launched from a directory that omits the
# project root from PYTHONPATH.
try:
    import session_manager  # noqa: F401

    logger.info("session_manager import successful")
except Exception as exc:  # pragma: no cover - log and continue
    logger.exception("session_manager import failed: %s", exc)


def _ensure_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        dir_path = os.path.dirname(file_path)
        listing = os.listdir(dir_path) if os.path.exists(dir_path) else []
        logger.error("File not found: %s. Dir contents: %s", file_path, listing)
        raise FileNotFoundError(f"Required file missing: {file_path}")


@app.task(bind=True, name="extract_problematic_accounts")
def extract_problematic_accounts(self, file_path: str, session_id: str | None = None):
    """Extract problematic accounts from the report."""
    try:
        logger.info("Extracting accounts from %s", file_path)
        _ensure_file(file_path)
        return extract_problematic_accounts_from_report(file_path, session_id)
    except Exception as exc:
        logger.exception("❌ Error extracting accounts")
        raise exc


@app.task(bind=True, name="process_report")
def process_report(
    self,
    file_path: str,
    email: str,
    goal: str = "Not specified",
    is_identity_theft: bool = False,
    session_id: str | None = None,
    structured_summaries: dict | None = None,
):
    """Process the SmartCredit report and email results.

    ``structured_summaries`` should contain only sanitized data extracted from
    the client's explanations. Raw text must never be passed into this task.
    """
    try:
        print("🔧 [Celery] process_report called!")
        print(f"📄 file_path: {file_path}")
        print(f"📧 email: {email}")
        print(f"🎯 goal: {goal}")
        print(f"🕵️ is_identity_theft: {is_identity_theft}")
        print(f"🆔 session_id: {session_id or '[none]'}")

        if not session_id:
            session_id = str(uuid.uuid4())

        _ensure_file(file_path)
        logger.info("Starting processing for %s", email)

        client_info = {
            "name": "Unknown",
            "address": "Unknown",
            "email": email,
            "goal": goal,
            "session_id": session_id,
            # Raw client notes are intentionally excluded from the processing
            # pipeline to prevent leakage into generated letters.
            "custom_dispute_notes": {},
            "structured_summaries": structured_summaries or {},
        }

        proofs_files = {"smartcredit_report": file_path}
        run_credit_repair_process(client_info, proofs_files, is_identity_theft)

        logger.info("Finished processing for %s", email)
        print("✅ [Celery] Finished processing")

    except Exception as exc:
        logger.exception("❌ Error processing report for %s", email)
        print(f"❌ [Celery] Exception: {exc}")
        raise exc
