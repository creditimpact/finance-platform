# ruff: noqa: E402
import ast
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
    maybe_run_ai_pipeline_task,
    run_validation_requirements_for_all_accounts,
)
from backend.runflow.decider import (
    StageStatus,
    decide_next,
    record_stage,
    reconcile_umbrella_barriers,
)
from backend.frontend.packs.generator import generate_frontend_packs_for_run
from backend.runflow.manifest import (
    update_manifest_frontend,
    update_manifest_state,
)
from backend.pipeline.runs import RunManifest, persist_manifest
from backend.core.logic.report_analysis.problem_case_builder import build_problem_cases
from backend.core.logic.report_analysis.problem_extractor import detect_problem_accounts
from backend.core.logic.report_analysis.text_provider import (
    extract_and_cache_text,
    load_cached_text,
)
from backend.core.logic.report_analysis.trace_cleanup import purge_after_export
from backend.core.config import ENABLE_VALIDATION_REQUIREMENTS
from backend.settings import PROJECT_ROOT
from backend.prevalidation.tasks import detect_and_persist_date_convention
from backend.core.runflow.env_snapshot import log_worker_env_snapshot

# Ensure the project root is always on sys.path so local modules can be
# imported even when the worker is launched from outside the repository
# directory.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from celery import Celery, shared_task, signals
from kombu import Queue

from backend.api.env_sanitize import sanitize_openai_env as _sanitize_openai_env

from backend.api.config import get_app_config
from backend.core.models import ClientInfo, ProofDocuments
from backend.core.orchestrators import run_credit_repair_process
from backend.core.utils.json_utils import _json_safe

_sanitize_openai_env()

app = Celery("tasks")

# Ensure note_style Celery tasks are registered when the worker boots.
import backend.ai.note_style.tasks  # noqa: E402,F401


def _default_queue_name() -> str:
    value = (os.getenv("CELERY_DEFAULT_QUEUE") or "").strip() or "celery"
    return value


def _frontend_queue_name() -> str:
    value = (os.getenv("CELERY_FRONTEND_QUEUE") or "").strip() or "frontend"
    return value


def _known_queue_names() -> list[str]:
    base: list[str] = [
        _default_queue_name(),
        _frontend_queue_name(),
        "merge",
        "validation",
        "note_style",
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for name in base:
        key = name.strip()
        if not key or key in seen:
            continue
        ordered.append(key)
        seen.add(key)
    existing = getattr(app.conf, "task_queues", None) or []
    for queue in existing:
        try:
            queue_name = getattr(queue, "name", None)
        except AttributeError:  # pragma: no cover - defensive
            queue_name = None
        if queue_name and queue_name not in seen:
            ordered.append(queue_name)
            seen.add(queue_name)
    return ordered


def _flatten_task_routes(routes: object) -> dict[str, dict]:
    mapping: dict[str, dict] = {}
    if isinstance(routes, dict):
        for task_name, route in routes.items():
            if isinstance(route, dict):
                mapping.setdefault(task_name, {}).update(route)
    elif isinstance(routes, (list, tuple)):
        for entry in routes:
            if isinstance(entry, dict):
                for task_name, route in entry.items():
                    if isinstance(route, dict):
                        mapping.setdefault(task_name, {}).update(route)
    return mapping


_TARGETED_TASK_ROUTES = {
    "backend.api.tasks.generate_frontend_packs_task": "frontend",
    "backend.ai.note_style.tasks.note_style_prepare_and_send_task": "note_style",
    "backend.ai.note_style.tasks.note_style_send_account_task": "note_style",
    "backend.ai.note_style.tasks.note_style_send_sid_task": "note_style",
    "backend.pipeline.auto_ai_tasks.validation_send": "validation",
}

_LAST_TARGETED_ROUTES: dict[str, str] = {}


def _ensure_frontend_queue_configuration() -> None:
    """Ensure Celery is aware of the dedicated frontend and note_style queues."""

    default_queue = _default_queue_name()
    frontend_queue = _frontend_queue_name()

    queues = list(getattr(app.conf, "task_queues", []) or [])
    known_names = _known_queue_names()
    existing_names = {getattr(queue, "name", None) for queue in queues}

    for name in known_names:
        if name not in existing_names:
            queues.append(Queue(name))
            existing_names.add(name)

    app.conf.task_create_missing_queues = True
    app.conf.task_default_queue = default_queue
    app.conf.task_default_exchange = os.getenv(
        "CELERY_DEFAULT_EXCHANGE", default_queue
    )
    app.conf.task_default_routing_key = os.getenv(
        "CELERY_DEFAULT_ROUTING_KEY", f"{default_queue}.default"
    )
    app.conf.task_queues = queues

    targeted_routes = dict(_TARGETED_TASK_ROUTES)
    targeted_routes["backend.api.tasks.generate_frontend_packs_task"] = frontend_queue

    global _LAST_TARGETED_ROUTES

    routes_config = getattr(app.conf, "task_routes", None)
    if not routes_config:
        app.conf.task_routes = {
            task_name: {"queue": queue_name, "routing_key": queue_name}
            for task_name, queue_name in targeted_routes.items()
        }
        _LAST_TARGETED_ROUTES = dict(targeted_routes)
        return

    if isinstance(routes_config, dict):
        for task_name, queue_name in targeted_routes.items():
            existing = routes_config.get(task_name)
            if isinstance(existing, dict):
                existing.setdefault("queue", queue_name)
                existing.setdefault("routing_key", queue_name)
            elif existing is None:
                routes_config[task_name] = {
                    "queue": queue_name,
                    "routing_key": queue_name,
                }
        app.conf.task_routes = routes_config
        _LAST_TARGETED_ROUTES = dict(targeted_routes)
        return

    if isinstance(routes_config, (list, tuple)):
        updated_routes = list(routes_config)
        for task_name, queue_name in targeted_routes.items():
            applied = False
            for entry in updated_routes:
                if not isinstance(entry, dict) or task_name not in entry:
                    continue
                value = entry.get(task_name)
                if isinstance(value, dict):
                    value.setdefault("queue", queue_name)
                    value.setdefault("routing_key", queue_name)
                    applied = True
                    break
            if not applied:
                updated_routes.append(
                    {task_name: {"queue": queue_name, "routing_key": queue_name}}
                )
        app.conf.task_routes = updated_routes
        _LAST_TARGETED_ROUTES = dict(targeted_routes)
        return

    # Fallback: leave exotic configurations untouched but append our routes.
    app.conf.task_routes = [
        routes_config,
        {
            task_name: {"queue": queue_name, "routing_key": queue_name}
            for task_name, queue_name in targeted_routes.items()
        },
    ]
    _LAST_TARGETED_ROUTES = dict(targeted_routes)


_ensure_frontend_queue_configuration()


def _log_active_queue_configuration() -> None:
    queues = getattr(app.conf, "task_queues", []) or []
    queue_descriptions: list[dict[str, object]] = []
    queue_names: list[str] = []
    for queue in queues:
        try:
            queue_name = getattr(queue, "name", None)
            exchange = getattr(queue, "exchange", None)
            routing_key = getattr(queue, "routing_key", None)
        except Exception as exc:  # pragma: no cover - defensive
            queue_descriptions.append({"repr": repr(queue), "error": str(exc)})
            continue
        if isinstance(queue_name, str):
            queue_names.append(queue_name)
        queue_descriptions.append(
            {
                "name": queue_name,
                "exchange": getattr(exchange, "name", exchange),
                "routing_key": routing_key,
            }
        )

    routes = _flatten_task_routes(getattr(app.conf, "task_routes", None))
    note_style_routes = {
        task_name: route
        for task_name, route in routes.items()
        if task_name.startswith("backend.ai.note_style.")
    }

    logger.info(
        "CELERY_QUEUE_ACTIVE default=%s frontend=%s known=%s queues=%s note_style_in_queues=%s",
        _default_queue_name(),
        _frontend_queue_name(),
        sorted(_known_queue_names()),
        queue_descriptions,
        "note_style" in queue_names,
    )
    if note_style_routes:
        logger.info("CELERY_NOTE_STYLE_ROUTES routes=%s", note_style_routes)
    else:
        logger.warning("CELERY_NOTE_STYLE_ROUTES routes=missing")

    if _LAST_TARGETED_ROUTES:
        logger.info("CELERY_TARGETED_ROUTES routes=%s", _LAST_TARGETED_ROUTES)
    else:
        logger.warning("CELERY_TARGETED_ROUTES routes=uninitialized")


def _parse_task_routes(raw: str) -> object | None:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None


@signals.worker_process_init.connect
def configure_worker(**_):
    log_worker_env_snapshot("celery_worker_init")
    try:
        cfg = get_app_config()
        os.environ.setdefault("OPENAI_API_KEY", cfg.ai.api_key)
        app.conf.update(
            broker_url=os.getenv("CELERY_BROKER_URL", cfg.celery_broker_url),
            result_backend=os.getenv("CELERY_RESULT_BACKEND", cfg.celery_broker_url),
        )
        _ensure_frontend_queue_configuration()

        routes_env = os.getenv("CELERY_TASK_ROUTES")
        if routes_env:
            parsed_routes = _parse_task_routes(routes_env)
            if parsed_routes is not None:
                logger.info("CELERY_TASK_ROUTES_ENV routes=%s", parsed_routes)
            else:
                logger.warning(
                    "CELERY_TASK_ROUTES_ENV_PARSE_FAILED raw=%s", routes_env
                )

        configured_routes = getattr(app.conf, "task_routes", None)
        if configured_routes:
            logger.info("CELERY_TASK_ROUTES_ACTIVE routes=%s", configured_routes)
        else:
            logger.info("CELERY_TASK_ROUTES_ACTIVE routes=none")

        _log_active_queue_configuration()
    except EnvironmentError as exc:
        logger.warning("Starting in parser-only mode: %s", exc)


# Configure logging to emit progress information from Celery workers.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)
log = logger
logger.info("OPENAI_API_KEY present=%s", bool(os.getenv("OPENAI_API_KEY")))
logging.getLogger("pdfminer").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*FontBBox.*")

_AUTO_AI_PIPELINE_ENQUEUED: set[str] = set()

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
def generate_frontend_packs_task(
    self,
    sid: str,
    *,
    runs_root: str | None = None,
    force: bool = False,
) -> dict:
    """Expose ``generate_frontend_packs_for_run`` as a Celery task."""

    return generate_frontend_packs_for_run(sid, runs_root=runs_root, force=force)


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

    manifest: RunManifest | None = None
    runs_root: Path | None = None
    manifest_load_failed = False

    def _ensure_manifest_root() -> Path | None:
        nonlocal manifest, runs_root, manifest_load_failed

        if runs_root is not None:
            return runs_root

        if manifest is None and not manifest_load_failed:
            try:
                manifest = RunManifest.for_sid(sid)
            except Exception:  # pragma: no cover - defensive logging
                log.error(
                    "MANIFEST_LOAD_FOR_RUNFLOW_FAILED sid=%s", sid, exc_info=True
                )
                manifest_load_failed = True
                return None

        if manifest is None:
            return None

        runs_root = manifest.path.parent.parent
        return runs_root

    summary = build_problem_cases(sid, candidates=candidates)
    cases_info = summary.get("cases", {}) if isinstance(summary, dict) else {}
    log.info(
        "CASES_BUILD_DONE sid=%s count=%s dir=%s",
        sid,
        cases_info.get("count"),
        cases_info.get("dir"),
    )

    try:
        detect_and_persist_date_convention(sid)
    except Exception:  # pragma: no cover - defensive logging
        log.error("DATE_CONVENTION_PIPELINE_FAILED sid=%s", sid, exc_info=True)

    if ENABLE_VALIDATION_REQUIREMENTS:
        try:
            stats = run_validation_requirements_for_all_accounts(sid)
        except Exception:  # pragma: no cover - defensive logging
            log.error(
                "VALIDATION_REQUIREMENTS_PIPELINE_FAILED sid=%s",
                sid,
                exc_info=True,
            )
        else:
            if isinstance(summary, dict):
                summary["validation_requirements"] = stats
            log.info(
                "VALIDATION_REQUIREMENTS_PIPELINE_DONE sid=%s processed=%s findings=%s",
                sid,
                stats.get("processed_accounts", 0),
                stats.get("findings", 0),
            )

            validation_ok = bool(stats.get("ok", True))
            stage_status: StageStatus = "built" if validation_ok else "error"
            findings_count = int(stats.get("findings_count", stats.get("findings", 0)) or 0)
            empty_ok = bool((stats.get("processed_accounts") or 0) == 0)
            notes_value = stats.get("notes")

            packs_total = int(stats.get("processed_accounts", 0) or 0)
            accounts_eligible = packs_total
            total_accounts = int(stats.get("total_accounts", 0) or 0)
            packs_skipped = max(0, total_accounts - accounts_eligible)
            validation_metrics = {
                "packs_total": packs_total,
                "accounts_eligible": accounts_eligible,
                "packs_skipped": packs_skipped,
            }

            runs_root = _ensure_manifest_root()

            record_stage(
                sid,
                "validation",
                status=stage_status,
                counts={"findings_count": findings_count},
                empty_ok=empty_ok,
                notes=notes_value,
                metrics=validation_metrics,
                runs_root=runs_root,
            )

    runs_root = _ensure_manifest_root()

    try:
        fe_result = generate_frontend_packs_for_run(sid, runs_root=runs_root)
    except Exception:  # pragma: no cover - defensive logging
        log.error("FRONTEND_PACKS_TASK_FAILED sid=%s", sid, exc_info=True)
        record_stage(
            sid,
            "frontend",
            status="error",
            counts={"packs_count": 0},
            empty_ok=True,
            notes="generation_failed",
            runs_root=runs_root,
        )
    else:
        if fe_result.get("autorun_disabled"):
            log.info(
                "FRONTEND_PACKS_AUTORUN_DISABLED sid=%s status=%s",
                sid,
                fe_result.get("status"),
            )
        else:
            packs_count = int(fe_result.get("packs_count", 0) or 0)
            frontend_status_value = str(fe_result.get("status") or "success")
            frontend_stage_status: StageStatus = (
                "error" if frontend_status_value == "error" else "published"
            )
            notes_override = (
                frontend_status_value
                if frontend_status_value not in {"", "success"}
                else None
            )
            empty_ok = bool(fe_result.get("empty_ok", packs_count == 0))

            record_stage(
                sid,
                "frontend",
                status=frontend_stage_status,
                counts={"packs_count": packs_count},
                empty_ok=empty_ok,
                notes=notes_override,
                runs_root=runs_root,
            )

            try:
                reconcile_umbrella_barriers(sid, runs_root=runs_root)
            except Exception:  # pragma: no cover - defensive logging
                logger.warning(
                    "FRONTEND_BARRIERS_RECONCILE_FAILED sid=%s", sid, exc_info=True
                )

            if frontend_stage_status == "success":
                manifest = update_manifest_frontend(
                    sid,
                    packs_dir=fe_result.get("packs_dir"),
                    packs_count=packs_count,
                    built=bool(fe_result.get("built", False)),
                    last_built_at=fe_result.get("last_built_at"),
                    manifest=manifest,
                    runs_root=runs_root,
                )

    decision = decide_next(sid, runs_root=runs_root)
    next_action = decision.get("next")

    next_action = decision.get("next") if next_action is None else next_action
    if next_action == "await_input":
        manifest = update_manifest_state(
            sid,
            "AWAITING_CUSTOMER_INPUT",
            manifest=manifest,
            runs_root=runs_root,
        )
    elif next_action == "complete_no_action":
        manifest = update_manifest_state(
            sid,
            "COMPLETE_NO_ACTION",
            manifest=manifest,
            runs_root=runs_root,
        )
    elif next_action == "stop_error":
        manifest = update_manifest_state(
            sid,
            "ERROR",
            manifest=manifest,
            runs_root=runs_root,
        )

    if os.environ.get("ENABLE_AUTO_AI_PIPELINE", "1") in ("1", "true", "True"):
        if has_ai_merge_best_tags(sid):
            if sid in _AUTO_AI_PIPELINE_ENQUEUED:
                log.info("AUTO_AI_ALREADY_ENQUEUED sid=%s", sid)
            else:
                try:
                    if manifest is None:
                        manifest = RunManifest.for_sid(sid)
                except Exception:
                    log.error("AUTO_AI_ENQUEUE_MANIFEST_FAILED sid=%s", sid, exc_info=True)
                else:
                    try:
                        manifest.set_ai_enqueued()
                        persist_manifest(manifest)
                        log.info("MANIFEST_AI_ENQUEUED sid=%s", sid)
                        maybe_run_ai_pipeline_task.delay(sid)
                    except Exception:
                        log.error("AUTO_AI_ENQUEUE_FAILED sid=%s", sid, exc_info=True)
                    else:
                        _AUTO_AI_PIPELINE_ENQUEUED.add(sid)
                        log.info("AUTO_AI_ENQUEUED sid=%s", sid)
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
