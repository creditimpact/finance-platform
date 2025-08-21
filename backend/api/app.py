# ruff: noqa: E402
import os
import sys

# Ensure the project root is always on sys.path, regardless of the
# working directory from which this module is executed.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import time
import uuid
from collections import defaultdict

from flask import Blueprint, Flask, jsonify, redirect, request, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.analytics.batch_runner import BatchFilters, BatchRunner
from backend.api.admin import admin_bp
from backend.api.auth import require_api_key_or_role
from backend.api.config import ENABLE_BATCH_RUNNER, get_app_config
from backend.api.session_manager import (
    get_session,
    set_session,
    update_intake,
    update_session,
)
from backend.api.tasks import run_credit_repair_process  # noqa: F401
from backend.api.tasks import (
    app as celery_app,
    extract_problematic_accounts,
    smoke_task,
)
from backend.core.logic.letters.explanations_normalizer import (
    extract_structured,
    sanitize,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


_request_counts: dict[str, list[float]] = defaultdict(list)


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

        original_name = uploaded_file.filename
        sanitized = secure_filename(original_name) or "report.pdf"
        ext = os.path.splitext(sanitized)[1]
        unique_name = f"{uuid.uuid4().hex}{ext}"
        local_filename = os.path.join(upload_folder, unique_name)
        uploaded_file.save(local_filename)

        if not os.path.exists(local_filename):
            logger.error("File failed to save at %s", local_filename)
            return jsonify({"status": "error", "message": "File upload failed"}), 500

        size = os.path.getsize(local_filename)
        print(f"File saved to {local_filename} ({size} bytes)")

        with open(local_filename, "rb") as f:
            first_bytes = f.read(4)
            print("First bytes of file:", first_bytes)
            if first_bytes != b"%PDF":
                print("File is not a valid PDF")
                return jsonify({"status": "error", "message": "Invalid PDF file"}), 400

        set_session(
            session_id,
            {
                "file_path": local_filename,
                "original_filename": original_name,
                "email": email,
            },
        )

        accounts = extract_problematic_accounts.delay(local_filename, session_id).get(
            timeout=300
        )

        legacy = {
            "negative_accounts": accounts.get("disputes", []),
            "open_accounts_with_issues": accounts.get("goodwill", []),
            "unauthorized_inquiries": accounts.get("inquiries", []),
            "high_utilization_accounts": accounts.get("high_utilization", []),
        }

        payload = {
            "status": "awaiting_user_explanations",
            "session_id": session_id,
            "filename": unique_name,
            "original_filename": original_name,
            "accounts": legacy,
        }

        logger.info("start_process payload: %s", payload)

        return jsonify(payload)

    except Exception as e:
        print("Exception occurred:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


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
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
    app.register_blueprint(admin_bp)
    app.register_blueprint(api_bp)

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


if __name__ == "__main__":  # pragma: no cover - manual execution
    debug_mode = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    create_app().run(host="0.0.0.0", port=5000, debug=debug_mode)
