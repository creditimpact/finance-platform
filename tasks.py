import os
import uuid
import logging
from celery import Celery
from dotenv import load_dotenv
from main import run_credit_repair_process

load_dotenv()

BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
app = Celery('tasks', broker=BROKER_URL, backend=BROKER_URL)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.task(bind=True, name='process_report')
def process_report(self, file_path: str, email: str, goal: str = "Not specified", is_identity_theft: bool = False, session_id: str | None = None):
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

        logger.info("Starting processing for %s", email)

        client_info = {
            "name": "Unknown",
            "address": "Unknown",
            "email": email,
            "goal": goal,
            "session_id": session_id,
        }

        proofs_files = {"smartcredit_report": file_path}
        run_credit_repair_process(client_info, proofs_files, is_identity_theft)

        logger.info("Finished processing for %s", email)
        print("✅ [Celery] Finished processing")

    except Exception as exc:
        logger.exception("❌ Error processing report for %s", email)
        print(f"❌ [Celery] Exception: {exc}")
        raise exc
