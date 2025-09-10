# ruff: noqa: E402
import json
import logging
import os
import sys
import uuid
import warnings
from pathlib import Path

from dotenv import load_dotenv

from backend.core.logic.report_analysis.extract_problematic_accounts import (
    extract_problematic_accounts as extract_problematic_accounts_logic,
)
from backend.core.logic.report_analysis.orchestrator import run_stage_a
from backend.core.logic.report_analysis.trace_cleanup import purge_after_export
from backend.settings import PROJECT_ROOT

# Ensure the project root is always on sys.path so local modules can be
# imported even when the worker is launched from outside the repository
# directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from celery import Celery, shared_task, signals

from backend.api.config import get_app_config
from backend.core.models import ClientInfo, ProofDocuments
from backend.core.orchestrators import run_credit_repair_process
from backend.core.utils.json_utils import _json_safe

app = Celery("tasks")


@signals.worker_process_init.connect
def configure_worker(**_):
    try:
        cfg = get_app_config()
        os.environ.setdefault("OPENAI_API_KEY", cfg.ai.api_key)
        app.conf.update(
            broker_url=os.getenv("CELERY_BROKER_URL", cfg.celery_broker_url),
            result_backend=os.getenv("CELERY_RESULT_BACKEND", cfg.celery_broker_url),
        )
    except EnvironmentError as exc:
        logger.warning("Starting in parser-only mode: %s", exc)


# Configure logging to emit progress information from Celery workers.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)
log = logger
logger.info("OPENAI_API_KEY present=%s", bool(os.getenv("OPENAI_API_KEY")))
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*FontBBox.*")

# Verify that session_manager is importable at startup. This helps catch
# cases where the worker is launched from a directory that omits the
# project root from PYTHONPATH.
try:
    from backend.api import session_manager  # noqa: F401

    logger.info("session_manager import successful")
except Exception as exc:  # pragma: no cover - log and continue
    logger.exception("session_manager import failed: %s", exc)


def _ensure_file(file_path: str) -> None:
    if not os.path.exists(file_path):
        dir_path = os.path.dirname(file_path)
        listing = os.listdir(dir_path) if os.path.exists(dir_path) else []
        logger.error("File not found: %s. Dir contents: %s", file_path, listing)
        raise FileNotFoundError(f"Required file missing: {file_path}")


@shared_task(bind=True)
def stage_a_task(self, sid: str) -> dict:
    """Run Stage-A export for the given session id."""
    log.info("STAGE_A start sid=%s", sid)
    result = run_stage_a(sid) or {}
    safe_result = _json_safe(result)
    try:
        json.dumps(safe_result, ensure_ascii=False)
    except TypeError as e:  # pragma: no cover - defensive logging
        logger.error("Non-JSON value at tasks.stage_a_task return: %s", e)
        raise
    log.info("STAGE_A done sid=%s", sid)
    return safe_result


@shared_task(bind=True)
def extract_problematic_accounts(self, sid: str) -> dict:
    """Extract problematic accounts for ``sid``.

    This task is defensive and does not rely on previous task outputs. It always
    returns a JSON-serializable dictionary.
    """
    log.info("PROBLEM_ACCOUNTS start sid=%s", sid)
    try:
        found = extract_problematic_accounts_logic(sid) or []
        result = {"sid": sid, "status": "ok", "found": found}
        safe_result = _json_safe(result)
        try:
            json.dumps(safe_result, ensure_ascii=False)
        except TypeError as e:
            logger.error(
                "Non-JSON value at tasks.extract_problematic_accounts return: %s",
                e,
            )
            raise
        return safe_result
    except Exception:
        logger.exception("PROBLEM_ACCOUNTS failed sid=%s", sid)
        raise


@app.task(bind=True, name="smoke_task")
def smoke_task(self):
    """Minimal task used for health checks."""
    return {"status": "ok"}


@shared_task(bind=True)
def cleanup_trace_task(self, sid: str) -> dict:
    """Purge parser traces for ``sid`` and keep final artifacts.

    Deletes everything under ``traces/blocks/<sid>`` except:
      - ``_debug_full.tsv``
      - ``accounts_from_full.json``
      - ``general_info_from_full.json``
    Also removes ``traces/texts/<sid>``. Always returns a dict to keep celery chains healthy.
    """
    log.info("TRACE_CLEANUP start sid=%s", sid)
    root = Path(PROJECT_ROOT)
    blocks_dir = root / "traces" / "blocks" / sid
    acct = blocks_dir / "accounts_table"
    must_exist = [
        "_debug_full.tsv",
        "accounts_from_full.json",
        "general_info_from_full.json",
    ]
    assert blocks_dir.exists(), f"blocks dir not found: {blocks_dir}"
    assert all((acct / k).exists() for k in must_exist), f"missing artifacts in {acct}"
    summary = purge_after_export(sid=sid, project_root=root)
    log.info(
        "TRACE_CLEANUP done sid=%s kept=['_debug_full.tsv','accounts_from_full.json','general_info_from_full.json']",
        sid,
    )
    result = {"sid": sid, "cleanup": summary}
    safe_result = _json_safe(result)
    try:
        json.dumps(safe_result, ensure_ascii=False)
    except TypeError as e:  # pragma: no cover - defensive logging
        logger.error("Non-JSON value at tasks.cleanup_trace_task return: %s", e)
        raise
    return safe_result


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
        print("[Celery] process_report called!")
        print(f"file_path: {file_path}")
        print(f"email: {email}")
        print(f"goal: {goal}")
        print(f"is_identity_theft: {is_identity_theft}")
        print(f"session_id: {session_id or '[none]'}")

        if not session_id:
            session_id = str(uuid.uuid4())

        _ensure_file(file_path)
        logger.info("Starting processing for %s", email)

        client = ClientInfo.from_dict(
            {
                "name": "Unknown",
                "address": "Unknown",
                "email": email,
                "goal": goal,
                "session_id": session_id,
                "structured_summaries": structured_summaries or {},
            }
        )

        proofs = ProofDocuments.from_dict({"smartcredit_report": file_path})
        run_credit_repair_process(client, proofs, is_identity_theft)

        logger.info("Finished processing for %s", email)
        print("[Celery] Finished processing")

    except Exception as exc:
        logger.exception("[ERROR] Error processing report for %s", email)
        print(f"[ERROR] [Celery] Exception: {exc}")
        raise exc
