from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
from tasks import process_report, extract_problematic_accounts
from admin import admin_bp

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
        local_filename = os.path.join(upload_folder, uploaded_file.filename)
        uploaded_file.save(local_filename)

        size = os.path.getsize(local_filename)
        print(f"💾 File saved to {local_filename} ({size} bytes)")

        with open(local_filename, "rb") as f:
            first_bytes = f.read(4)
            print("🧪 First bytes of file:", first_bytes)
            if first_bytes != b"%PDF":
                print("❌ File is not a valid PDF")
                return jsonify({"status": "error", "message": "Invalid PDF file"}), 400

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
            "filename": uploaded_file.filename,
            "accounts": accounts,
        })

    except Exception as e:
        print("❌ Exception occurred:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/submit-explanations", methods=["POST"])
def submit_explanations():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    filename = data.get("filename")
    email = data.get("email")
    goal = data.get("goal", "Not specified")
    is_identity_theft = data.get("is_identity_theft", False)
    explanations = data.get("explanations", {})

    if not (session_id and filename and email):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    file_path = os.path.join("uploads", session_id, filename)
    process_report.delay(file_path, email, goal, is_identity_theft, session_id, explanations)

    return jsonify({"status": "processing", "message": "Letters generation started."})

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
