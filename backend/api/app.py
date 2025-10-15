# ruff: noqa: E402
import os
import re
import sys

# Ensure the project root is always on sys.path, regardless of the
# working directory from which this module is executed.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

from backend.api.env_sanitize import sanitize_openai_env

load_dotenv()

sanitize_openai_env()

import json
import logging
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any, Mapping

from flask import Blueprint, Flask, jsonify, redirect, request, url_for
from flask_cors import CORS
from jsonschema import Draft7Validator, ValidationError

import backend.config as config
from backend.analytics.batch_runner import BatchFilters, BatchRunner
from backend.api.admin import admin_bp
from backend.api.ai_endpoints import ai_bp
from backend.api.auth import require_api_key_or_role
from backend.api.config import ENABLE_BATCH_RUNNER, get_app_config
from backend.api.pipeline import run_full_pipeline
from backend.api.session_manager import (
    get_session,
    set_session,
    update_intake,
    update_session,
)
from backend.api.tasks import run_credit_repair_process  # noqa: F401
from backend.api.tasks import app as celery_app, smoke_task
from backend.pipeline.runs import RunManifest, persist_manifest
from backend.frontend.packs.responses import append_frontend_response
from backend.api.routes_smoke import bp as smoke_bp
from backend.api.ui_events import ui_event_bp
from backend.core import orchestrators as orch
from backend.core.case_store import api as cs_api
from backend.core.case_store.errors import NOT_FOUND, CaseStoreError
from backend.core.collectors import (
    collect_stageA_logical_accounts,
    collect_stageA_problem_accounts,
)
from backend.core.config.flags import FLAGS
from backend.core.logic.letters.explanations_normalizer import (
    extract_structured,
    sanitize,
)
from backend.core.materialize.casestore_view import build_account_view

logger = logging.getLogger(__name__)
log = logger

api_bp = Blueprint("api", __name__)


SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
with open(SCHEMA_DIR / "problem_account.json") as _f:
    _problem_account_validator = Draft7Validator(json.load(_f))


_request_counts: dict[str, list[float]] = defaultdict(list)


def _runs_root_path() -> Path:
    root = os.getenv("RUNS_ROOT")
    return Path(root) if root else Path("runs")


def _validate_sid(sid: str) -> str:
    sid = (sid or "").strip()
    if not sid or sid.startswith("/") or sid.startswith("\\"):
        raise ValueError("invalid sid")
    if "/" in sid or "\\" in sid:
        raise ValueError("invalid sid")
    parts = Path(sid).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("invalid sid")
    return sid


def _run_dir_for_sid(sid: str) -> Path:
    validated = _validate_sid(sid)
    return _runs_root_path() / validated


def _frontend_stage_dir(run_dir: Path) -> Path:
    stage_dir_env = os.getenv("FRONTEND_PACKS_STAGE_DIR", "frontend/review")
    return run_dir / Path(stage_dir_env)


def _frontend_stage_index_candidates(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    stage_default = _frontend_stage_dir(run_dir) / "index.json"
    candidates.append(stage_default)

    index_env = os.getenv("FRONTEND_PACKS_INDEX")
    if index_env:
        env_candidate = Path(index_env)
        if not env_candidate.is_absolute():
            env_candidate = run_dir / env_candidate
        if env_candidate not in candidates:
            candidates.append(env_candidate)

    legacy_index = run_dir / "frontend" / "index.json"
    if legacy_index not in candidates:
        candidates.append(legacy_index)
    return candidates


def _frontend_stage_packs_dir(run_dir: Path) -> Path:
    packs_dir_env = os.getenv("FRONTEND_PACKS_DIR")
    if packs_dir_env:
        return run_dir / Path(packs_dir_env)
    return _frontend_stage_dir(run_dir) / "packs"


def _candidate_account_keys(account_id: str) -> list[str]:
    trimmed = (account_id or "").strip()
    candidates: list[str] = []
    if trimmed:
        candidates.append(trimmed)
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", trimmed)
    if sanitized and sanitized not in candidates:
        candidates.append(sanitized)
    return candidates


def _safe_relative_path(run_dir: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        return rel

    candidate = (run_dir / rel).resolve(strict=False)
    try:
        base = run_dir.resolve(strict=False)
    except FileNotFoundError:
        base = run_dir

    if base == candidate or base in candidate.parents:
        return candidate

    raise ValueError("path escapes run directory")


def _load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _merge_collectors(
    problems: list[Mapping[str, Any]] | None,
    logical: list[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    """Merge Stage-A collector outputs while dropping parser artifacts."""
    merged: dict[tuple[str | None, str | None], dict] = {}
    for acc in problems or []:
        key = (acc.get("account_id"), acc.get("bureau"))
        merged[key] = dict(acc)
    for acc in logical or []:
        key = (acc.get("account_id"), acc.get("bureau"))
        if key in merged:
            merged[key].update(acc)
        else:
            merged[key] = dict(acc)
    result: list[Mapping[str, Any]] = []
    for acc in merged.values():
        acc.pop("source_stage", None)
        result.append(acc)
    return result


@api_bp.route("/")
def index():
    return jsonify({"status": "ok", "message": "API is up"})


@api_bp.route("/api/smoke", methods=["GET"])
def smoke():
    """Lightweight health check verifying Celery round-trip."""
    result = smoke_task.delay().get(timeout=10)
    return jsonify({"ok": True, "celery": result})


@api_bp.route("/api/batch-runner", methods=["POST"])
@require_api_key_or_role(roles={"batch_runner"})
def run_batch_job():
    if not ENABLE_BATCH_RUNNER:
        return jsonify({"status": "error", "message": "batch runner disabled"}), 403
    data = request.get_json(force=True)
    filters_data = data.get("filters", {}) or {}
    action_tags = filters_data.get("action_tags")
    if not action_tags:
        return (
            jsonify({"status": "error", "message": "action_tags required"}),
            400,
        )

    cycle_range = filters_data.get("cycle_range")
    if isinstance(cycle_range, list):
        cycle_range = tuple(cycle_range)  # type: ignore[assignment]

    filters = BatchFilters(
        action_tags=action_tags,
        family_ids=filters_data.get("family_ids"),
        cycle_range=cycle_range,
        start_ts=filters_data.get("start_ts"),
        end_ts=filters_data.get("end_ts"),
        page_size=filters_data.get("page_size"),
        page_token=filters_data.get("page_token"),
    )

    fmt = data.get("format", "json")
    runner = BatchRunner()
    job_id = runner.run(filters, fmt)
    return jsonify({"status": "ok", "job_id": job_id})


def _load_frontend_stage_manifest(run_dir: Path) -> tuple[Path, Any] | None:
    for candidate in _frontend_stage_index_candidates(run_dir):
        if not candidate.is_file():
            continue
        try:
            payload = _load_json_file(candidate)
        except json.JSONDecodeError:
            logger.warning(
                "FRONTEND_STAGE_INDEX_DECODE_FAILED path=%s", candidate, exc_info=True
            )
            continue
        except OSError:
            logger.warning(
                "FRONTEND_STAGE_INDEX_READ_FAILED path=%s", candidate, exc_info=True
            )
            continue

        return candidate, payload
    return None


def _load_frontend_pack(pack_path: Path) -> Any:
    try:
        return _load_json_file(pack_path)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "FRONTEND_PACK_DECODE_FAILED path=%s error=%s", pack_path, exc, exc_info=True
        )
        raise
    except OSError as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "FRONTEND_PACK_READ_FAILED path=%s error=%s", pack_path, exc, exc_info=True
        )
        raise


@api_bp.route("/api/runs/<sid>/frontend/review/index", methods=["GET"])
def api_frontend_review_index(sid: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    manifest = _load_frontend_stage_manifest(run_dir)
    if manifest is None:
        return jsonify({"error": "not_found"}), 404

    _, payload = manifest
    return jsonify(payload)


def _stage_pack_path_for_account(run_dir: Path, account_id: str) -> Path | None:
    stage_dir = _frontend_stage_packs_dir(run_dir)
    for key in _candidate_account_keys(account_id):
        candidate = stage_dir / f"{key}.json"
        if candidate.is_file():
            return candidate
    return None


def _legacy_pack_path_for_account(run_dir: Path, account_id: str) -> Path | None:
    base = run_dir / "frontend" / "accounts"
    for key in _candidate_account_keys(account_id):
        candidate = base / key / "pack.json"
        if candidate.is_file():
            return candidate
    return None


@api_bp.route("/api/runs/<sid>/frontend/review/accounts/<account_id>", methods=["GET"])
def api_frontend_review_pack(sid: str, account_id: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    stage_pack = _stage_pack_path_for_account(run_dir, account_id)
    if stage_pack is None:
        manifest_info = _load_frontend_stage_manifest(run_dir)
        if manifest_info:
            _, manifest_payload = manifest_info
            packs = manifest_payload.get("packs") if isinstance(manifest_payload, Mapping) else None
            if isinstance(packs, list):
                account_keys = set(_candidate_account_keys(account_id))
                for pack_meta in packs:
                    if not isinstance(pack_meta, Mapping):
                        continue
                    meta_account_id = pack_meta.get("account_id")
                    if isinstance(meta_account_id, str) and meta_account_id in account_keys:
                        path_value = pack_meta.get("path")
                        if isinstance(path_value, str):
                            try:
                                candidate = _safe_relative_path(run_dir, path_value)
                            except ValueError:
                                continue
                            if candidate.is_file():
                                stage_pack = candidate
                                break

    if stage_pack is None:
        stage_pack = _legacy_pack_path_for_account(run_dir, account_id)

    if stage_pack is None or not stage_pack.is_file():
        return jsonify({"error": "pack_not_found"}), 404

    try:
        payload = _load_frontend_pack(stage_pack)
    except Exception:  # pragma: no cover - error path
        return jsonify({"error": "pack_read_failed"}), 500

    return jsonify(payload)


@api_bp.route(
    "/api/runs/<sid>/frontend/review/accounts/<account_id>/answer",
    methods=["POST"],
)
def api_frontend_review_answer(sid: str, account_id: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    if not run_dir.exists():
        return jsonify({"error": "run_not_found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    if not isinstance(data, Mapping):
        return jsonify({"error": "invalid_payload"}), 400

    answers = data.get("answers")
    if not isinstance(answers, Mapping):
        return jsonify({"error": "invalid_answers"}), 400

    client_ts = data.get("client_ts")
    if client_ts is not None and not isinstance(client_ts, str):
        return jsonify({"error": "invalid_client_ts"}), 400

    record: dict[str, Any] = {
        "sid": sid,
        "account_id": account_id,
        "answers": dict(answers),
        "received_at": _now_utc_iso(),
    }
    if client_ts is not None:
        record["client_ts"] = client_ts

    append_frontend_response(run_dir, account_id, record)

    return jsonify({"ok": True})


@api_bp.route("/api/start-process", methods=["POST"])
def start_process():
    try:
        print("Received request to /api/start-process")

        form = request.form or {}
        uploaded_file = request.files.get("file")
        email = form.get("email")
        if not uploaded_file:
            return jsonify({"status": "error", "message": "Missing file"}), 400

        session_id = str(uuid.uuid4())
        upload_folder = os.path.join("uploads", session_id)
        os.makedirs(upload_folder, exist_ok=True)

        original_name = uploaded_file.filename or "report.pdf"
        pdf_path = Path(upload_folder) / "smartcredit_report.pdf"
        uploaded_file.save(pdf_path)

        if not pdf_path.exists():
            logger.error("File failed to save at %s", pdf_path)
            return (
                jsonify({"status": "error", "message": "File upload failed"}),
                400,
            )

        size = pdf_path.stat().st_size
        print(f"File saved to {pdf_path} ({size} bytes)")

        with open(pdf_path, "rb") as f:
            first_bytes = f.read(4)
            print("First bytes of file:", first_bytes)
            if first_bytes != b"%PDF":
                print("File is not a valid PDF")
                return jsonify({"status": "error", "message": "Invalid PDF file"}), 400

        manifest = RunManifest.for_sid(session_id)
        uploads_dir = manifest.ensure_run_subdir("uploads_dir", "uploads")
        dst = (uploads_dir / pdf_path.name).resolve()
        if pdf_path.resolve() != dst:
            copy2(pdf_path, dst)
        manifest.set_artifact("uploads", "smartcredit_report", dst)
        persist_manifest(manifest)
        log.info(
            "MANIFEST_ARTIFACT_ADDED sid=%s group=%s name=%s path=%s",
            manifest.sid,
            "uploads",
            "smartcredit_report",
            dst,
        )

        set_session(
            session_id,
            {
                "file_path": str(dst),
                "original_filename": original_name,
                "email": email,
            },
        )

        if not pdf_path.exists():
            return (
                jsonify({"status": "error", "message": "PDF missing"}),
                400,
            )

        result = run_full_pipeline(session_id).get(timeout=300)

        try:
            cs_api.load_session_case(session_id)
            problem_accounts = orch.collect_stageA_logical_accounts(session_id)
        except CaseStoreError:
            logger.exception("casestore_unavailable session=%s", session_id)
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Case Store unavailable",
                    }
                ),
                503,
            )

        valid_accounts = []
        for acc in problem_accounts:
            to_validate = dict(acc)
            to_validate.pop("aggregation_meta", None)
            try:
                _problem_account_validator.validate(to_validate)
                valid_accounts.append(acc)
            except ValidationError:
                logger.warning(
                    "invalid_problem_account session=%s account=%s",
                    session_id,
                    acc,
                    exc_info=True,
                )
        problem_accounts = valid_accounts
        if config.API_INCLUDE_DECISION_META:
            for acc in problem_accounts:
                meta = orch.get_stageA_decision_meta(session_id, acc.get("account_id"))
                if meta is None:
                    meta = {
                        "decision_source": acc.get("decision_source", "rules"),
                        "confidence": acc.get("confidence", 0.0),
                        "tier": acc.get("tier", "none"),
                    }
                    fields_used = acc.get("fields_used")
                    if fields_used:
                        meta["fields_used"] = fields_used
                fields_used = meta.get("fields_used")
                if fields_used:
                    meta["fields_used"] = list(fields_used)[
                        : config.API_DECISION_META_MAX_FIELDS_USED
                    ]
                acc["decision_meta"] = meta

        legacy = request.args.get("legacy", "").lower() in ("1", "true", "yes")

        accounts = {
            # Primary field
            "problem_accounts": problem_accounts,
        }

        if legacy:
            # Backward compatibility fields for legacy clients
            accounts["negative_accounts"] = problem_accounts
            accounts["open_accounts_with_issues"] = problem_accounts

        accounts["unauthorized_inquiries"] = result.get(
            "unauthorized_inquiries", result.get("inquiries", [])
        )
        accounts["high_utilization_accounts"] = result.get(
            "high_utilization_accounts", result.get("high_utilization", [])
        )

        payload = {
            "status": "awaiting_user_explanations",
            "session_id": session_id,
            "filename": pdf_path.name,
            "original_filename": original_name,
            "accounts": accounts,
        }

        logger.info("start_process payload: %s", payload)

        return jsonify(payload)

    except Exception as e:
        print("Exception occurred:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Async upload → queue analysis
# ---------------------------------------------------------------------------


@api_bp.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        form = request.form or {}
        email = (form.get("email") or "").strip()
        file = request.files.get("file")
        if not email or not file:
            return jsonify({"ok": False, "message": "missing fields"}), 400

        session_id = str(uuid.uuid4())
        upload_folder = os.path.join("uploads", session_id)
        os.makedirs(upload_folder, exist_ok=True)

        original_name = file.filename or "report.pdf"
        pdf_path = Path(upload_folder) / "smartcredit_report.pdf"
        file.save(pdf_path)

        # Basic PDF validation
        with open(pdf_path, "rb") as f:
            if f.read(4) != b"%PDF":
                return jsonify({"ok": False, "message": "Invalid PDF file"}), 400

        if not pdf_path.exists():
            return jsonify({"ok": False, "message": "File upload failed"}), 400

        manifest = RunManifest.for_sid(session_id)
        uploads_dir = manifest.ensure_run_subdir("uploads_dir", "uploads")
        dst = (uploads_dir / pdf_path.name).resolve()
        if pdf_path.resolve() != dst:
            copy2(pdf_path, dst)
        manifest.set_artifact("uploads", "smartcredit_report", dst)
        persist_manifest(manifest)

        # Persist initial session state
        set_session(
            session_id,
            {
                "file_path": str(dst),
                "original_filename": original_name,
                "email": email,
                "status": "queued",
            },
        )

        # Queue background extraction (non-blocking)
        task = run_full_pipeline(session_id)
        update_session(session_id, task_id=task.id, status="queued")

        # Return explicit async contract (frontend polls /api/result)
        return (
            jsonify(
                {
                    "ok": True,
                    "status": "queued",
                    "session_id": session_id,
                    "task_id": task.id,
                }
            ),
            202,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("upload failed")
        return jsonify({"ok": False, "message": str(e)}), 500


@api_bp.route("/api/result", methods=["GET"])
def api_result():
    session_id = (request.args.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"ok": False, "message": "missing session_id"}), 400
    session = get_session(session_id)
    if session is None:
        # Tolerant contract: treat not-found as in-progress to avoid noisy 404s
        return jsonify({"ok": True, "status": "processing"}), 200

    status = session.get("status") or "queued"
    if status in ("queued", "processing"):
        return jsonify({"ok": True, "status": status}), 200
    if status == "error":
        return (
            jsonify(
                {"ok": False, "status": "error", "message": session.get("error") or ""}
            ),
            200,
        )

    # done
    payload = session.get("result") or {}
    return (
        jsonify(
            {
                "ok": True,
                "status": "done",
                "session_id": session_id,
                "result": payload,
            }
        ),
        200,
    )


@api_bp.route("/api/explanations", methods=["POST"])
def explanations_endpoint():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    explanations = data.get("explanations", [])

    if not session_id or not isinstance(explanations, list):
        return jsonify({"status": "error", "message": "Invalid input"}), 400

    structured: list[dict] = []
    raw_store: list[dict] = []
    for item in explanations:
        text = item.get("text", "")
        ctx = {
            "account_id": item.get("account_id", ""),
            "dispute_type": item.get("dispute_type", ""),
        }
        raw_store.append({"account_id": ctx["account_id"], "text": text})
        safe = sanitize(text)
        structured.append(extract_structured(safe, ctx))

    update_session(session_id, structured_summaries=structured)
    update_intake(session_id, raw_explanations=raw_store)
    return jsonify({"status": "ok", "structured": structured})


@api_bp.route("/api/summaries/<session_id>", methods=["GET"])
def get_summaries(session_id: str):
    session = get_session(session_id)
    if not session:
        return jsonify({"status": "error", "message": "Session not found"}), 404
    raw = session.get("structured_summaries", {}) or {}
    allowed = {
        "account_id",
        "dispute_type",
        "facts_summary",
        "claimed_errors",
        "dates",
        "evidence",
        "risk_flags",
    }
    cleaned: dict[str, dict] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                key = item.get("account_id") or str(len(cleaned))
                cleaned[key] = {k: item.get(k) for k in allowed if k in item}
    elif isinstance(raw, dict):
        for key, item in raw.items():
            cleaned[key] = {k: item.get(k) for k in allowed if k in item}
    logger.debug("summaries payload for %s: %s", session_id, cleaned)
    return jsonify({"status": "ok", "summaries": cleaned})


@api_bp.route("/api/account-transitions/<session_id>/<account_id>", methods=["GET"])
def account_transitions(session_id: str, account_id: str):
    session = get_session(session_id)
    if not session:
        return jsonify({"status": "error", "message": "Session not found"}), 404
    states = session.get("account_states", {}) or {}
    data = states.get(str(account_id))
    if not data:
        return jsonify({"status": "error", "message": "Account not found"}), 404
    return jsonify({"status": "ok", "history": data.get("history", [])})


@api_bp.route("/api/submit-explanations", methods=["POST"])
def submit_explanations():
    return redirect(url_for("api.explanations_endpoint"), code=307)


def create_app() -> Flask:
    sanitize_openai_env()
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(ai_bp)
    app.register_blueprint(ui_event_bp)
    app.register_blueprint(smoke_bp, url_prefix="/smoke")

    @app.before_request
    def _load_config() -> None:
        if getattr(app, "_config_loaded", False):
            return
        cfg = get_app_config()
        celery_app.conf.update(
            broker_url=cfg.celery_broker_url,
            result_backend=cfg.celery_broker_url,
        )
        os.environ.setdefault("OPENAI_API_KEY", cfg.ai.api_key)
        app.secret_key = cfg.secret_key
        app.auth_tokens = cfg.auth_tokens
        app.rate_limit_per_minute = cfg.rate_limit_per_minute
        logger.info("Flask app starting with OPENAI_BASE_URL=%s", cfg.ai.base_url)
        logger.info("Flask app OPENAI_API_KEY present=%s", bool(cfg.ai.api_key))
        app._config_loaded = True

    @app.before_request
    def _auth_and_throttle() -> tuple[dict, int] | None:
        tokens: list[str] = getattr(app, "auth_tokens", [])
        limit: int = getattr(app, "rate_limit_per_minute", 60)
        identifier = request.remote_addr or "global"
        if tokens:
            auth_header = request.headers.get("Authorization", "")
            token = (
                auth_header[7:].strip() if auth_header.startswith("Bearer ") else None
            )
            if token not in tokens:
                return jsonify({"status": "error", "message": "Unauthorized"}), 401
            identifier = token
        now = time.time()
        recent = [t for t in _request_counts[identifier] if now - t < 60]
        if len(recent) >= limit:
            return jsonify({"status": "error", "message": "Too Many Requests"}), 429
        recent.append(now)
        _request_counts[identifier] = recent

    return app


# ---------------------------------------------------------------------------
# Accounts API (reads analyzer-produced artifacts)
# ---------------------------------------------------------------------------


@api_bp.route("/api/account/<session_id>/<account_id>", methods=["GET"])
def account_view_api(session_id: str, account_id: str):
    if not session_id or not account_id:
        return jsonify({"error": "invalid_request"}), 400
    try:
        view = build_account_view(session_id, account_id)
    except CaseStoreError as exc:  # pragma: no cover - error path
        if getattr(exc, "code", "") == NOT_FOUND:
            return jsonify({"error": "account_not_found"}), 404
        logger.exception(
            "account_view_failed session=%s account=%s", session_id, account_id
        )
        return jsonify({"error": "internal_error"}), 500
    return jsonify(view)


@api_bp.route("/api/accounts/<session_id>", methods=["GET"])
def list_accounts_api(session_id: str):
    """Return compact list of problem accounts built from Case Store artifacts."""

    if not session_id:
        return jsonify({"ok": False, "message": "missing session_id"}), 400

    try:
        probs = collect_stageA_problem_accounts(session_id) or []
    except CaseStoreError:
        probs = []
    try:
        logical = collect_stageA_logical_accounts(session_id) or []
    except CaseStoreError:
        logical = []

    accounts = _merge_collectors(probs, logical)

    if FLAGS.case_first_build_required and not accounts:
        return jsonify({"ok": True, "session_id": session_id, "accounts": []})

    return jsonify({"ok": True, "session_id": session_id, "accounts": accounts})


@api_bp.route("/api/problem_accounts")
def api_problem_accounts_legacy():
    """Legacy parser-first endpoint intentionally disabled."""
    if FLAGS.disable_parser_ui_summary:
        return jsonify({"ok": False, "error": "parser_first_disabled"}), 410
    return jsonify({"ok": False, "error": "parser_first_disabled"}), 410


@api_bp.route("/api/cases/<session_id>", methods=["GET"])
def api_list_cases(session_id: str):
    try:
        session_case = cs_api.load_session_case(session_id)
    except Exception as e:  # pragma: no cover - debug endpoint
        return (
            jsonify({"ok": False, "session_id": session_id, "error": str(e)}),
            200,
        )

    accounts = session_case.accounts or {}
    logical_index = session_case.summary.logical_index or {}
    reverse_index = {aid: lk for lk, aid in logical_index.items()}
    items = []
    for aid, account in accounts.items():
        issuer = None
        logical_key = reverse_index.get(aid)
        try:
            by_bureau = getattr(account.fields, "by_bureau", {}) or {}
            for bureau_code in ("EX", "EQ", "TU"):
                bureau_obj = by_bureau.get(bureau_code) or {}
                issuer = (
                    issuer
                    or bureau_obj.get("issuer")
                    or bureau_obj.get("creditor_name")
                )
        except Exception:
            pass
        items.append({"case_id": aid, "issuer": issuer, "logical_key": logical_key})

    return jsonify({"ok": True, "session_id": session_id, "cases": items})


@api_bp.route("/api/session/<session_id>/logical_index", methods=["GET"])
def api_logical_index(session_id: str):
    try:
        session_case = cs_api.load_session_case(session_id)
        idx = session_case.summary.logical_index or {}
        return jsonify({"ok": True, "session_id": session_id, "logical_index": idx})
    except Exception as e:  # pragma: no cover - debug endpoint
        return (
            jsonify({"ok": False, "session_id": session_id, "error": str(e)}),
            200,
        )


if __name__ == "__main__":  # pragma: no cover - manual execution
    debug_mode = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    create_app().run(host="0.0.0.0", port=5000, debug=debug_mode)
