import os
import sys

# Ensure the project root is always on sys.path, regardless of the
# working directory from which this module is executed.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import config
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
from werkzeug.utils import secure_filename

from tasks import process_report, extract_problematic_accounts
from admin import admin_bp
from session_manager import set_session, get_session, update_session
from logic.explanations_normalizer import sanitize, extract_structured

logger = logging.getLogger(__name__)
logger.info("Flask app starting with OPENAI_BASE_URL=%s", config.OPENAI_BASE_URL)
logger.info("Flask app OPENAI_API_KEY present=%s", bool(config.OPENAI_API_KEY))

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

app.secret_key = os.environ.get("SECRET_KEY", "change-me")
app.register_blueprint(admin_bp)

@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "API is up"})

@app.route("/api/start-process", methods=["POST"])
def start_process():
    try:
        print("🔔 Received request to /api/start-process")

        data = request.get_json(silent=True)
        print("📦 Raw body:", request.data[:200])
        print("📑 Headers:", dict(request.headers))
        print("🧾 Parsed JSON:", data)

        if not data:
            data = request.form
            print("📤 Using form-data:", data)

        uploaded_file = request.files.get("file")
        print(f"📧 Email: {data.get('email')}")
        print(f"📎 Uploaded file: {uploaded_file.filename if uploaded_file else '❌ None'}")

        if not data.get("email") or not uploaded_file:
            return jsonify({"status": "error", "message": "Missing email or file"}), 400

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
        print(f"💾 File saved to {local_filename} ({size} bytes)")

        with open(local_filename, "rb") as f:
            first_bytes = f.read(4)
            print("🧪 First bytes of file:", first_bytes)
            if first_bytes != b"%PDF":
                print("❌ File is not a valid PDF")
                return jsonify({"status": "error", "message": "Invalid PDF file"}), 400

        set_session(session_id, {
            "file_path": local_filename,
            "original_filename": original_name,
            "email": data.get("email")
        })

        goal = data.get("goal")
        is_identity_theft = str(data.get("is_identity_theft", "")).lower() in ("1", "true", "yes")
        legal_name = data.get("legal_name")
        address = data.get("address")
        story = data.get("story")
        notes = data.get("custom_dispute_notes")

        print("📋 Collected fields:", {
            "goal": goal,
            "legal_name": legal_name,
            "address": address,
            "story": story,
            "is_identity_theft": is_identity_theft,
            "custom_dispute_notes": notes
        })

        accounts = extract_problematic_accounts.delay(local_filename, session_id).get(timeout=300)

        return jsonify({
            "status": "ok",
            "session_id": session_id,
            "filename": unique_name,
            "original_filename": original_name,
            "accounts": accounts,
        })

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/explanations", methods=["POST"])
def explanations_endpoint():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    explanations = data.get("explanations", [])

    if not session_id or not isinstance(explanations, list):
        return jsonify({"status": "error", "message": "Invalid input"}), 400

    structured: list[dict] = []
    for item in explanations:
        text = item.get("text", "")
        ctx = {
            "account_id": item.get("account_id", ""),
            "dispute_type": item.get("dispute_type", ""),
        }
        safe = sanitize(text)
        structured.append(extract_structured(safe, ctx))

    update_session(session_id, structured_summaries=structured)
    return jsonify({"status": "ok", "structured": structured})


@app.route("/api/submit-explanations", methods=["POST"])
def submit_explanations():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    email = data.get("email")
    goal = data.get("goal", "Not specified")
    is_identity_theft = data.get("is_identity_theft", False)
    explanations = data.get("explanations", {})

    if not (session_id and email):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    session = get_session(session_id)
    file_path = session.get("file_path") if session else None
    if not file_path or not os.path.exists(file_path):
        dir_listing = []
        if session:
            dir_path = os.path.dirname(file_path)
            if os.path.exists(dir_path):
                dir_listing = os.listdir(dir_path)
        logger.error("File for session %s missing at %s. Dir contents: %s", session_id, file_path, dir_listing)
        return jsonify({"status": "error", "message": "Uploaded report is missing. Please restart the process."}), 404

    process_report.delay(file_path, email, goal, is_identity_theft, session_id, explanations)

    return jsonify({"status": "processing", "message": "Letters generation started."})

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
