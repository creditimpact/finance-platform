# ruff: noqa: E402
import json
import shutil
import logging
import os
import sys
import uuid
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from backend.core.logic.report_analysis.block_exporter import export_stage_a, run_stage_a
from backend.pipeline.auto_ai import (
    has_ai_merge_best_tags,
    maybe_run_ai_pipeline,
)
from backend.pipeline.runs import RunManifest
from backend.core.logic.report_analysis.problem_case_builder import build_problem_cases
from backend.core.logic.report_analysis.problem_extractor import detect_problem_accounts
from backend.core.logic.report_analysis.text_provider import (
    extract_and_cache_text,
    load_cached_text,
)
from backend.core.logic.report_analysis.trace_cleanup import purge_after_export
from backend.settings import PROJECT_ROOT

# Ensure the project root is always on sys.path so local modules can be
# imported even when the worker is launched from outside the repository
# directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    root = Path(PROJECT_ROOT)
    uploads = root / "uploads" / sid
    pdf = uploads / "smartcredit_report.pdf"
    if not pdf.exists():
        cands = list(uploads.glob("*.pdf"))
        if cands:
            pdf = cands[0]
    if not pdf.exists():
        log.error("stage_a_task: PDF missing under %s", uploads)
        result = {
            "sid": sid,
            "ok": False,
            "where": "stage_a",
            "reason": "pdf_missing",
            "uploads_dir": str(uploads),
        }
        safe_result = _json_safe(result)
        try:
            json.dumps(safe_result, ensure_ascii=False)
        except TypeError as e:  # pragma: no cover - defensive logging
            logger.error("Non-JSON value at tasks.stage_a_task return: %s", e)
            raise
        log.info("STAGE_A end sid=%s", sid)
        return safe_result
    try:
        cached = load_cached_text(sid)
        have = bool(cached and cached.get("pages"))
    except Exception:
        have = False
    if not have:
        ocr_on = os.getenv("OCR_ENABLED", "0") == "1"
        extract_and_cache_text(session_id=sid, pdf_path=str(pdf), ocr_enabled=ocr_on)
        log.info("TEXT_CACHE built sid=%s", sid)
    try:
        # Enforce canonical Stage-A output dir under runs/<SID>/traces/accounts_table
        m = RunManifest.for_sid(sid)
        traces_dir = m.ensure_run_subdir("traces_dir", "traces")
        accounts_out_dir = (traces_dir / "accounts_table").resolve()
        accounts_out_dir.mkdir(parents=True, exist_ok=True)
        # Record canonical accounts_table base dir in manifest
        m.set_base_dir("traces_accounts_table", accounts_out_dir)
        # Guardrail: ensure we never write to legacy traces/blocks
        assert "runs" in str(accounts_out_dir), "Stage-A out_dir must live under runs/<SID>"
        logger.info("STAGE_A_CANONICAL_OUT sid=%s dir=%s", sid, accounts_out_dir); run_stage_a(sid=sid, accounts_out_dir=accounts_out_dir); m.set_artifact("traces.accounts_table","accounts_json", accounts_out_dir / "accounts_from_full.json"); m.set_artifact("traces.accounts_table","general_json", accounts_out_dir / "general_info_from_full.json"); m.set_artifact("traces.accounts_table","debug_full_tsv", accounts_out_dir / "_debug_full.tsv"); m.set_artifact("traces.accounts_table","per_account_tsv_dir", accounts_out_dir / "per_account_tsv")
        # Defensive: auto-sync any legacy traces into the canonical runs/<SID>
        try:
            from scripts.sync_traces_into_runs import sync_one as _sync_legacy

            _sync_legacy(sid, move=True)
        except Exception as e_sync:
            logger.warning("SYNC_LEGACY_TRACES_FAILED sid=%s err=%s", sid, e_sync)
    except Exception as e:
        log.exception("export_stage_a_failed")
        result = {
            "sid": sid,
            "ok": False,
            "where": "stage_a",
            "reason": "export_failed",
            "error": str(e),
        }
        safe_result = _json_safe(result)
        try:
            json.dumps(safe_result, ensure_ascii=False)
        except TypeError as e2:  # pragma: no cover - defensive logging
            logger.error("Non-JSON value at tasks.stage_a_task return: %s", e2)
            raise
        log.info("STAGE_A end sid=%s", sid)
        return safe_result
    result = {"sid": sid, "ok": True, "where": "stage_a"}
    safe_result = _json_safe(result)
    try:
        json.dumps(safe_result, ensure_ascii=False)
    except TypeError as e:  # pragma: no cover - defensive logging
        logger.error("Non-JSON value at tasks.stage_a_task return: %s", e)
        raise
    log.info("STAGE_A end sid=%s", sid)
    return safe_result


def _manifest_keep_set(m: RunManifest) -> set[Path]:
    # Build allow-list of files to keep from manifest
    keep: set[Path] = set()
    acct_json = Path(m.get("traces.accounts_table", "accounts_json")).resolve()
    gen_json = Path(m.get("traces.accounts_table", "general_json")).resolve()
    debug_tsv = Path(m.get("traces.accounts_table", "debug_full_tsv")).resolve()
    per_dir = Path(m.get("traces.accounts_table", "per_account_tsv_dir")).resolve()

    keep.update({acct_json, gen_json, debug_tsv})
    if per_dir.exists():
        if per_dir.is_dir():
            for p in per_dir.rglob("*"):
                if p.is_file():
                    keep.add(p.resolve())
            keep.add(per_dir.resolve())
        else:
            keep.add(per_dir.resolve())

    return keep

@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def extract_problematic_accounts(self, sid: str) -> dict:
    """Analyze Stage-A accounts and return candidates only (no writes)."""
    log.info("PROBLEMATIC start sid=%s", sid)
    found = detect_problem_accounts(sid)
    log.info("PROBLEMATIC done sid=%s found=%d", sid, len(found))
    result = {"sid": sid, "problematic": len(found), "found": found}
    # Optional auto-build cases
    if os.getenv("ANALYZER_AUTO_BUILD_CASES", "1") == "1":
        try:
            build_problem_cases_task.delay(result, sid)  # type: ignore[name-defined]
        except Exception:
            log.warning("AUTO_BUILD_CASES_FAILED sid=%s", sid, exc_info=True)
    return result


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def build_problem_cases_task(self, prev: dict | None = None, sid: str | None = None) -> dict:
    """Create per-account case folders for problematic candidates under runs/<SID>/cases/accounts.

    When chained after extract_problematic_accounts, ``prev`` receives the previous result.
    """
    # Resolve SID and candidates
    if sid is None and isinstance(prev, dict):
        sid = str(prev.get("sid"))
    assert sid, "sid is required"
    candidates = []
    if isinstance(prev, dict) and prev.get("found"):
        candidates = list(prev.get("found") or [])
    else:
        candidates = detect_problem_accounts(sid)

    summary = build_problem_cases(sid, candidates=candidates)
    cases_info = summary.get("cases", {}) if isinstance(summary, dict) else {}
    log.info(
        "CASES_BUILD_DONE sid=%s count=%s dir=%s",
        sid,
        cases_info.get("count"),
        cases_info.get("dir"),
    )

    if os.environ.get("ENABLE_AUTO_AI_PIPELINE", "1") in ("1", "true", "True"):
        try:
            manifest = RunManifest.for_sid(sid)
        except Exception:
            log.error("AUTO_AI_ENQUEUE_MANIFEST_FAILED sid=%s", sid, exc_info=True)
        else:
            runs_root = manifest.path.parent.parent
            if has_ai_merge_best_tags(runs_root, sid):
                try:
                    maybe_run_ai_pipeline.delay(sid)
                    log.info("AUTO_AI_ENQUEUED sid=%s", sid)
                except Exception:
                    log.error("AUTO_AI_ENQUEUE_FAILED sid=%s", sid, exc_info=True)
            else:
                log.info("AUTO_AI_SKIP_NO_CANDIDATES sid=%s", sid)

    return summary


@app.task(bind=True, name="smoke_task")
def smoke_task(self):
    """Minimal task used for health checks."""
    return {"status": "ok"}


@shared_task(bind=True)
def cleanup_trace_task(self, sid: str) -> dict:
    """Canonical cleanup using manifest allow-list under runs/<SID>/traces.

    - Operates under runs/<SID>/traces
    - Keeps only manifest-listed Stage-A artifacts (incl. per_account_tsv/*)
    - Optionally removes legacy traces/blocks/<SID> if canonical artifacts exist
    """
    m = RunManifest.for_sid(sid)
    traces_dir = m.ensure_run_subdir("traces_dir", "traces").resolve()
    accounts_dir = (traces_dir / "accounts_table").resolve()

    log.info("TRACE_CLEANUP_CANONICAL sid=%s base=%s", sid, traces_dir)

    # Build allow-list
    try:
        keep = _manifest_keep_set(m)
    except Exception as e:
        log.warning(
            "TRACE_CLEANUP_SKIP sid=%s reason=manifest_missing_keys err=%s", sid, e
        )
        return {"sid": sid, "cleanup": {"performed": False, "reason": "manifest_missing_keys"}}

    # Collect all files under runs/<SID>/traces
    all_files = [p.resolve() for p in traces_dir.rglob("*") if p.is_file()]

    # Compute delete candidates = files not in keep
    to_delete = [p for p in all_files if p not in keep]

    # Delete files
    deleted: list[str] = []
    for p in to_delete:
        try:
            p.unlink()
            deleted.append(str(p))
        except FileNotFoundError:
            continue
        except Exception as e:
            log.warning("TRACE_CLEANUP_DELETE_FAIL sid=%s file=%s err=%s", sid, p, e)

    # Best-effort prune of empty directories (avoid removing accounts_dir and base traces_dir)
    to_check_dirs = sorted({p.parent for p in to_delete}, key=lambda x: len(str(x)), reverse=True)
    for d in to_check_dirs:
        if d == accounts_dir or d == traces_dir:
            continue
        try:
            if d.exists() and d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        except Exception:
            pass

    # Optionally remove legacy traces/blocks/<SID> if canonical artifacts exist
    legacy = Path("traces") / "blocks" / sid
    legacy_removed = False
    required = [
        accounts_dir / "_debug_full.tsv",
        accounts_dir / "accounts_from_full.json",
        accounts_dir / "general_info_from_full.json",
    ]
    if legacy.exists() and all(p.exists() for p in required):
        try:
            shutil.rmtree(legacy)
            legacy_removed = True
        except Exception as e:
            log.warning(
                "TRACE_CLEANUP_LEGACY_REMOVE_FAIL sid=%s dir=%s err=%s", sid, legacy, e
            )

    kept_list = [str(p) for p in keep if p.exists()]
    log.info(
        "TRACE_CLEANUP_DONE sid=%s kept=%d deleted=%d legacy_removed=%s",
        sid,
        len(kept_list),
        len(deleted),
        legacy_removed,
    )

    return {
        "sid": sid,
        "cleanup": {
            "performed": True,
            "canonical_base": str(traces_dir),
            "kept": kept_list,
            "deleted": deleted,
            "legacy_removed": legacy_removed,
        },
    }


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
