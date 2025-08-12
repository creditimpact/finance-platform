"""High level interface for generating instruction PDFs.

This module provides the public function ``generate_instruction_file`` which
coordinates data preparation and HTML rendering before producing the final PDF
and JSON context.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from backend.core.logic.instruction_data_preparation import prepare_instruction_data
from backend.core.services.ai_client import AIClient
from backend.core.logic.instruction_renderer import build_instruction_html
from backend.core.logic import pdf_renderer
from backend.core.logic.compliance_pipeline import run_compliance_pipeline
from backend.core.models import ClientInfo, BureauPayload
from backend.assets.paths import templates_path


def get_logo_base64() -> str:
    """Return the Credit Impact logo encoded as a base64 data URI."""
    logo_path = Path(templates_path("Logo_CreditImpact.png"))
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    return ""


def generate_instruction_file(
    client: ClientInfo,
    bureau_map: Mapping[str, BureauPayload | dict[str, Any]],
    is_identity_theft: bool,
    output_path: Path,
    ai_client: AIClient,
    run_date: str | None = None,
    strategy: dict | None = None,
    wkhtmltopdf_path: str | None = None,
):
    """Generate the instruction PDF and JSON context for the client."""
    run_date = run_date or datetime.now().strftime("%B %d, %Y")
    logo_base64 = get_logo_base64()

    if isinstance(client, dict):  # pragma: no cover - backward compat
        client = ClientInfo.from_dict(client)
    client_info = client.to_dict()
    bureau_data = {
        k: (
            (
                BureauPayload.from_dict(v).to_dict()
                if isinstance(v, dict)
                else v.to_dict()
            )
            if isinstance(v, (BureauPayload, dict))
            else v
        )
        for k, v in bureau_map.items()
    }

    context, all_accounts = prepare_instruction_data(
        client_info,
        bureau_data,
        is_identity_theft,
        run_date,
        logo_base64,
        ai_client=ai_client,
        strategy=strategy,
    )
    html = build_instruction_html(context)
    run_compliance_pipeline(
        html,
        client_info.get("state"),
        client_info.get("session_id", ""),
        "instructions",
        ai_client=ai_client,
    )

    if wkhtmltopdf_path:
        render_pdf_from_html(html, output_path, wkhtmltopdf_path=wkhtmltopdf_path)
    else:
        render_pdf_from_html(html, output_path)
    save_json_output(all_accounts, output_path)

    print("[✅] Instructions file generated successfully.")


def render_pdf_from_html(
    html: str, output_path: Path, wkhtmltopdf_path: str | None = None
) -> Path:
    """Persist the rendered PDF to disk."""
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "Start_Here - Instructions.pdf"
    if wkhtmltopdf_path:
        pdf_renderer.render_html_to_pdf(
            html, str(filepath), wkhtmltopdf_path=wkhtmltopdf_path
        )
    else:
        pdf_renderer.render_html_to_pdf(html, str(filepath))
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
