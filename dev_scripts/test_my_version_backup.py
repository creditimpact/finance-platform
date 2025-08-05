import os
import sys
from pathlib import Path

# הוספת הנתיב למודולים
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import run_credit_repair_process

# 💼 פרטי לקוח - Alirio Exposito Sr
client_info = {
    "name": "Alirio Exposito Sr",
    "legal_name": "Alirio Exposito Sr",
    "email": "alirio.test@example.com",
    "address": "58 N BLUFF CREEK CIR, THE WOODLANDS, TX 77382",
    "state": "TX",
    "is_identity_theft": False,
    "session_id": "alirio_exposito_test_01",
    "custom_dispute_notes": {
        "BMW FINANCIAL": "I didn’t make the first three payments because I thought auto-pay was set up. By the time I noticed, it was too late. I’ve paid everything since and ask for consideration."
    }
}

# 🔍 מיקום הדוח SmartCredit
smartcredit_path = "uploads/smartcredit_report.pdf"  # ודא שהדוח כבר נמצא שם
if not os.path.exists(smartcredit_path):
    raise FileNotFoundError(f"❌ Missing SmartCredit report at: {smartcredit_path}")

proofs_files = {
    "smartcredit_report": smartcredit_path
}

# 🚀 הרצת הבדיקה
if __name__ == "__main__":
    print("🚀 Running full test for Alirio Exposito Sr with strategy and custom notes...")
    run_credit_repair_process(client_info, proofs_files, is_identity_theft=False)
    print("✅ Test complete! Check output folder and logs.")
