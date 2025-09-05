from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PyPDF2 import PdfReader

MAX_UPLOAD_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".pdf"}


def is_valid_filename(file_path: Path) -> bool:
    return file_path.name.replace(" ", "").isalnum() or file_path.name.endswith(".pdf")


def contains_suspicious_pdf_elements(file_path: Path) -> bool:
    try:
        with open(file_path, "rb") as f:
            content = f.read().lower()
            suspicious_keywords = [
                b"/js",
                b"/javascript",
                b"/launch",
                b"/aa",
                b"/openaction",
            ]
            return any(keyword in content for keyword in suspicious_keywords)
    except Exception as e:
        print(f"[⚠️] Failed to scan for PDF threats: {e}")
        return True


def is_safe_pdf(file_path: Path) -> bool:
    print(f"[INFO] Checking PDF: {file_path.name}")

    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        print(f"[✗] Blocked: Unsupported file extension {file_path.suffix}")
        return False

    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        print(f"[✗] Blocked: File size {size_mb:.2f} MB exceeds {MAX_UPLOAD_SIZE_MB} MB")
        return False

    suspicious = contains_suspicious_pdf_elements(file_path)
    print(f"[INFO] Suspicious markers found: {suspicious}")
    if suspicious:
        print("[⚠️] Suspicious PDF markers detected but not blocking")

    try:
        reader = PdfReader(str(file_path))
        page_count = len(reader.pages)
        print(f"[INFO] Pages found: {page_count}")
    except Exception as e:
        print(f"[✗] Failed to open PDF: {e}")
        return False

    print("[✅] PDF passed all checks.")
    return True


def move_uploaded_file(uploaded_path: Path, session_id: str) -> Path:
    safe_folder = Path("uploads") / session_id
    safe_folder.mkdir(parents=True, exist_ok=True)

    original_name_path = safe_folder / uploaded_path.name
    uploaded_path.replace(original_name_path)

    standard_path = safe_folder / "smartcredit_report.pdf"
    original_name_path.replace(standard_path)

    return standard_path
