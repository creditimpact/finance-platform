"""High level interface for generating instruction PDFs.

This module provides the public function ``generate_instruction_file`` which
coordinates data preparation and HTML rendering before producing the final PDF
and JSON context.
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from pathlib import Path

import pdfkit

from logic.instruction_data_preparation import (
    prepare_instruction_data,
    generate_account_action,  # re-exported for backward compatibility
)
from logic.instruction_renderer import build_instruction_html


def get_logo_base64() -> str:
    """Return the Credit Impact logo encoded as a base64 data URI."""
    logo_path = Path("templates/Logo_CreditImpact.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    return ""


def render_html_to_pdf(html_string: str, output_path: Path):
    """Render the provided HTML string to a PDF file."""
    config = pdfkit.configuration(
        wkhtmltopdf=os.getenv("WKHTMLTOPDF_PATH", "wkhtmltopdf")
    )
    options = {"quiet": ""}
    try:
        pdfkit.from_string(html_string, str(output_path), configuration=config, options=options)
        print(f"[📄] PDF rendered: {output_path}")
    except Exception as e:
        print(f"[❌] Failed to render PDF: {e}")


def generate_html(
    client_info,
    bureau_data,
    is_identity_theft: bool,
    run_date: str,
    logo_base64: str,
    strategy: dict | None = None,
):
    """Return the rendered HTML and merged account list.

    This is kept for backward compatibility with modules that previously
    imported :func:`generate_html` directly.
    """
    context, all_accounts = prepare_instruction_data(
        client_info,
        bureau_data,
        is_identity_theft,
        run_date,
        logo_base64,
        strategy,
    )
    html = build_instruction_html(context)
    return html, all_accounts


def generate_instruction_file(
    client_info,
    bureau_data,
    is_identity_theft: bool,
    output_path: Path,
    run_date: str | None = None,
    strategy: dict | None = None,
):
    """Generate the instruction PDF and JSON context for the client."""
    run_date = run_date or datetime.now().strftime("%B %d, %Y")
    logo_base64 = get_logo_base64()

    context, all_accounts = prepare_instruction_data(
        client_info,
        bureau_data,
        is_identity_theft,
        run_date,
        logo_base64,
        strategy,
    )

    html = build_instruction_html(context)

    render_pdf_from_html(html, output_path)
    save_json_output(all_accounts, output_path)

    print("[✅] Instructions file generated successfully.")


def render_pdf_from_html(html: str, output_path: Path) -> Path:
    """Persist the rendered PDF to disk."""
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "Start_Here - Instructions.pdf"
    render_html_to_pdf(html, filepath)
    return filepath


def save_json_output(all_accounts: list[dict], output_path: Path):
    """Write the sanitized account context to a JSON file."""
    def sanitize_for_json(data):
        if isinstance(data, dict):
            return {k: sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [sanitize_for_json(i) for i in data]
        elif isinstance(data, set):
            return list(data)
        else:
            return data

    sanitized_accounts = sanitize_for_json(all_accounts)
    with open(output_path / "instructions_context.json", "w") as f:
        json.dump({"accounts": sanitized_accounts}, f, indent=2)
