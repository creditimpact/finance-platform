"""PDF text extraction and conversion helpers."""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF
import pdfplumber
import fitz


def convert_txts_to_pdfs(folder: Path):
    """Converts .txt files in the given folder to styled PDFs with Unicode support."""
    txt_files = list(folder.glob("*.txt"))
    output_folder = folder / "converted"
    output_folder.mkdir(exist_ok=True)

    for txt_path in txt_files:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        font_path = "fonts/DejaVuSans.ttf"
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.add_font("DejaVu", "B", font_path, uni=True)
        pdf.set_font("DejaVu", "B", 14)

        title = txt_path.stem
        pdf.cell(0, 10, title, ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("DejaVu", "", 12)

        with open(txt_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    pdf.ln(5)
                    continue
                try:
                    pdf.multi_cell(0, 10, line)
                except Exception as e:
                    print(f"[âš ï¸] Failed to render line: {line[:50]} â€" {e}")
                    continue

        new_path = output_folder / (txt_path.stem + ".pdf")
        pdf.output(str(new_path))
        print(f"[ðŸ"„] Converted to PDF: {new_path}")


def extract_pdf_text_safe(pdf_path: Path, max_chars: int = 4000) -> str:
    """Extract text from a PDF using pdfplumber with a fitz fallback."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            parts = []
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text:
                    parts.append(text)
                if sum(len(p) for p in parts) >= max_chars:
                    break
            joined = "\n".join(parts)
            if joined:
                return joined[:max_chars]
    except Exception as e:
        print(f"[âš ï¸] pdfplumber failed for {pdf_path}: {e}")

    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) >= max_chars:
                break
        doc.close()
        return text[:max_chars]
    except Exception as e:
        print(f"[âŒ] Fallback extraction failed for {pdf_path}: {e}")
        return ""


def gather_supporting_docs(
    session_id: str, max_chars: int = 4000
) -> tuple[str, list[str], dict[str, str]]:
    """Return a summary text, list of filenames and mapping of snippets for supplemental PDFs."""
    base = Path("supporting_docs")
    candidates = []
    if session_id:
        candidates.append(base / session_id)
    candidates.append(base)

    summaries = []
    filenames = []
    doc_snippets: dict[str, str] = {}
    total_len = 0

    for folder in candidates:
        if not folder.exists():
            continue
        for pdf_path in sorted(folder.glob("*.pdf")):
            if total_len >= max_chars:
                print("[âš ï¸] Reached max characters, truncating remaining docs.")
                break
            try:
                raw_text = extract_pdf_text_safe(pdf_path, 1500)
                snippet = " ".join(raw_text.split())[:700] if raw_text else ""
                if snippet:
                    summary = (
                        f"The following document was provided: '{pdf_path.name}'\n"
                        f"â†' Summary: {snippet}"
                    )
                    summaries.append(summary)
                    doc_snippets[pdf_path.name] = snippet
                    total_len += len(summary) + 1
                filenames.append(pdf_path.name)
                print(f"[ðŸ"Ž] Parsed supporting doc: {pdf_path.name}")
            except Exception as e:
                print(f"[âš ï¸] Failed to parse {pdf_path.name}: {e}")
                continue
        if total_len >= max_chars:
            break

    combined = "\n".join(summaries)
    if len(combined) > max_chars:
        combined = combined[:max_chars]
        print("[âš ï¸] Combined supporting docs summary truncated due to length.")

    return combined.strip(), filenames, doc_snippets


def gather_supporting_docs_text(session_id: str, max_chars: int = 4000) -> str:
    """Backward compatible wrapper returning only the summary text."""
    summary, _, _ = gather_supporting_docs(session_id, max_chars)
    return summary
