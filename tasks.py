import os
import uuid
import logging
import config
from celery import Celery
from main import run_credit_repair_process, extract_problematic_accounts_from_report

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
app = Celery('tasks', broker=BROKER_URL, backend=BROKER_URL)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Celery worker starting with OPENAI_BASE_URL=%s", config.OPENAI_BASE_URL)
logger.info("Celery worker OPENAI_API_KEY present=%s", bool(config.OPENAI_API_KEY))


def _ensure_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        dir_path = os.path.dirname(file_path)
        listing = os.listdir(dir_path) if os.path.exists(dir_path) else []
        logger.error("File not found: %s. Dir contents: %s", file_path, listing)
        raise FileNotFoundError(f"Required file missing: {file_path}")

@app.task(bind=True, name='extract_problematic_accounts')
def extract_problematic_accounts(self, file_path: str, session_id: str | None = None):
    """Extract problematic accounts from the report."""
    try:
        logger.info("Extracting accounts from %s", file_path)
        _ensure_file(file_path)
        return extract_problematic_accounts_from_report(file_path, session_id)
    except Exception as exc:
        logger.exception("❌ Error extracting accounts")
        raise exc


@app.task(bind=True, name='process_report')
def process_report(self, file_path: str, email: str, goal: str = "Not specified", is_identity_theft: bool = False, session_id: str | None = None, explanations: dict | None = None):
    """Process the SmartCredit report and email results."""
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
            "custom_dispute_notes": explanations or {},
        }

        proofs_files = {"smartcredit_report": file_path}
        run_credit_repair_process(client_info, proofs_files, is_identity_theft)

        logger.info("Finished processing for %s", email)
        print("✅ [Celery] Finished processing")

    except Exception as exc:
        logger.exception("❌ Error processing report for %s", email)
        print(f"❌ [Celery] Exception: {exc}")
        raise exc
