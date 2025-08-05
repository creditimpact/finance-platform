import os
import requests
from datetime import datetime


def send_admin_login_alert(ip: str | None = None) -> None:
    """Send Telegram message when admin logs in."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"\u26a0\ufe0f Admin panel login detected!\nTime: {timestamp}"
    if ip:
        text += f"\nIP: {ip}"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=5)
    except Exception:
        pass
