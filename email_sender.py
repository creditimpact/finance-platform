import smtplib
import os
from email.message import EmailMessage
from pathlib import Path

def collect_all_files(folder: Path):
    """אוסף את כל קבצי ה־PDF מתוך תיקיית הלקוח לשליחה באימייל."""
    return [str(p) for p in folder.glob("*.pdf") if p.is_file()]

def send_email_with_attachment(receiver_email, subject, body, files):
    smtp_server = os.getenv("SMTP_SERVER", "localhost")
    smtp_port = int(os.getenv("SMTP_PORT", "1025"))
    sender_email = os.getenv("SMTP_USERNAME", "noreply@example.com")
    sender_password = os.getenv("SMTP_PASSWORD", "")  # local dev default

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    for file_path in files:
        if not os.path.exists(file_path):
            print(f"[⚠️] File not found, skipping: {file_path}")
            continue

        with open(file_path, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(file_path)
            msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)

    with smtplib.SMTP(smtp_server, smtp_port) as smtp:
        try:
            smtp.starttls()
            if sender_password:
                smtp.login(sender_email, sender_password)
        except Exception:
            pass
        smtp.send_message(msg)
        print(f"📧 Email sent to {receiver_email}")
