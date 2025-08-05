import os
from pathlib import Path
import pdfplumber

MIN_TEXT_CHARS = 300

MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".pdf"}

def is_valid_filename(file_path: Path) -> bool:
    """בודק אם שם הקובץ לא מכיל תווים חשודים"""
    return file_path.name.replace(" ", "").isalnum() or file_path.name.endswith(".pdf")

def contains_suspicious_pdf_elements(file_path: Path) -> bool:
    """בודק אם יש קוד זדוני ב־PDF (JS, Launch וכו׳)"""
    try:
        with open(file_path, "rb") as f:
            content = f.read().lower()
            suspicious_keywords = [b"/js", b"/javascript", b"/launch", b"/aa", b"/openaction"]
            return any(keyword in content for keyword in suspicious_keywords)
    except Exception as e:
        print(f"[⚠️] Failed to scan for PDF threats: {e}")
        return True


def is_safe_pdf(file_path: Path) -> bool:
    """Validate the PDF for size, basic safety and readable text."""
    print(f"[INFO] Checking PDF: {file_path.name}")

    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        print(f"[❌] Blocked: Unsupported file extension {file_path.suffix}")
        return False

    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        print(f"[❌] Blocked: File size {size_mb:.2f} MB exceeds {MAX_UPLOAD_SIZE_MB} MB")
        return False

    suspicious = contains_suspicious_pdf_elements(file_path)
    print(f"[INFO] Suspicious markers found: {suspicious}")
    if suspicious:
        print("[⚠️] Suspicious PDF markers detected but not blocking")

    try:
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            print(f"[INFO] Pages found: {page_count}")
            text_parts = []
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
            combined_text = "\n".join(text_parts)
            char_count = len(combined_text)
            print(f"[INFO] Total extracted characters: {char_count}")

            if char_count == 0:
                print("[❌] Blocked: No readable text detected")
                return False

            first_page_chars = len(text_parts[0]) if text_parts else 0
            if char_count < MIN_TEXT_CHARS:
                print("[❌] Blocked: Too little readable text (<300 chars)")
                return False
            if first_page_chars < 50 and char_count >= MIN_TEXT_CHARS:
                print("[INFO] First page short but total text sufficient")

    except Exception as e:
        print(f"[❌] Failed to open PDF: {e}")
        return False

    print("[✅] PDF passed all checks.")
    return True

def move_uploaded_file(uploaded_path: Path, session_id: str) -> Path:
    """מעביר את הקובץ לתיקייה ייחודית לפי session ID + שומר גם בשם המקורי"""
    safe_folder = Path("uploads") / session_id
    safe_folder.mkdir(parents=True, exist_ok=True)

    # שימור שם מקורי
    original_name_path = safe_folder / uploaded_path.name
    uploaded_path.replace(original_name_path)

    # העתקה כפולה בשם אחיד אם רוצים להשתמש בשם קבוע גם בהמשך
    standard_path = safe_folder / "smartcredit_report.pdf"
    original_name_path.replace(standard_path)

    return standard_path
