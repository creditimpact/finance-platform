import os
import json
import logging
import config
from pathlib import Path
from datetime import datetime

def save_analytics_snapshot(client_info: dict, report_summary: dict):
    logging.getLogger(__name__).info("Analytics tracker using OPENAI_BASE_URL=%s", config.OPENAI_BASE_URL)
    analytics_dir = Path("analytics_data")
    analytics_dir.mkdir(exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M")

    filename = analytics_dir / f"{timestamp}.json"

    snapshot = {
        "date": now.strftime("%Y-%m-%d"),
        "goal": client_info.get("goal", "N/A"),
        "dispute_type": "identity_theft" if client_info.get("is_identity_theft") else "standard",
        "client_name": client_info.get("name", "Unknown"),
        "client_state": client_info.get("state", "unknown"),
        "summary": {
            "num_collections": report_summary.get("num_collections", 0),
            "num_late_payments": report_summary.get("num_late_payments", 0),
            "high_utilization": report_summary.get("high_utilization", False),
            "recent_inquiries": report_summary.get("recent_inquiries", 0),
            "total_inquiries": report_summary.get("total_inquiries", 0),
            "num_negative_accounts": report_summary.get("num_negative_accounts", 0),
            "num_accounts_over_90_util": report_summary.get("num_accounts_over_90_util", 0),
            "account_types_in_problem": report_summary.get("account_types_in_problem", [])
        },
        "strategic_recommendations": report_summary.get("strategic_recommendations", [])
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"[📊] Analytics snapshot saved: {filename}")
