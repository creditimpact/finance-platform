from datetime import datetime
import config  # Ensures environment variables are loaded when this script runs


def send_admin_login_alert(ip: str | None = None) -> None:
    """Log admin login events locally instead of sending network requests."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"⚠️ Admin panel login detected! Time: {timestamp}"
    if ip:
        msg += f" | IP: {ip}"
    print(msg)
