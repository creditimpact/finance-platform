"""Rendering helpers for goodwill letters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from audit import AuditLogger, AuditLevel
from services.ai_client import AIClient

from logic.pdf_renderer import ensure_template_env, render_html_to_pdf as default_pdf_renderer
from logic.utils.file_paths import safe_filename
from logic.utils.note_handling import get_client_address_lines
from logic.utils.names_normalization import normalize_creditor_name
from logic.compliance_pipeline import run_compliance_pipeline as default_compliance


_template = ensure_template_env().get_template("goodwill_letter_template.html")


def load_creditor_address_map() -> Dict[str, str]:
    """Load a mapping of normalized creditor names to addresses."""
    try:
        with open("data/creditor_addresses.json", encoding="utf-8") as f:
            raw = json.load(f)
            if isinstance(raw, list):
                return {
                    normalize_creditor_name(entry["name"]): entry["address"]
                    for entry in raw
                    if "name" in entry and "address" in entry
                }
            if isinstance(raw, dict):
                return {normalize_creditor_name(k): v for k, v in raw.items()}
            print("[⚠️] Unknown address file format.")
            return {}
    except Exception as e:  # pragma: no cover - file IO issues
        print(f"[❌] Failed to load creditor addresses: {e}")
        return {}


def render_goodwill_letter(
    creditor: str,
    gpt_data: Dict[str, Any],
    client_info: Dict[str, Any],
    output_path: Path,
    run_date: str | None = None,
    *,
    doc_names: List[str] | None = None,
    ai_client: AIClient | None = None,
    audit: AuditLogger | None = None,
    compliance_fn=default_compliance,
    pdf_fn=default_pdf_renderer,
) -> None:
    """Render a goodwill letter using ``gpt_data`` and save a PDF and JSON."""

    client_name = client_info.get("legal_name") or client_info.get("name", "Your Name")
    if not client_info.get("legal_name"):
        print("[⚠️] Warning: legal_name not found in client_info. Using fallback name.")

    date_str = run_date or datetime.now().strftime("%B %d, %Y")
    address_map = load_creditor_address_map()
    creditor_key = normalize_creditor_name(creditor)
    creditor_address = address_map.get(creditor_key)
    if not creditor_address:
        print(f"[⚠️] No address found for: {creditor}")
        creditor_address = "Address not provided — please enter manually"

    session_id = client_info.get("session_id") or ""

    context = {
        "date": date_str,
        "client_name": client_name,
        "client_address_lines": get_client_address_lines(client_info),
        "creditor": creditor,
        "creditor_address": creditor_address,
        "accounts": gpt_data.get("accounts", []),
        "intro_paragraph": gpt_data.get("intro_paragraph", ""),
        "hardship_paragraph": gpt_data.get("hardship_paragraph", ""),
        "recovery_paragraph": gpt_data.get("recovery_paragraph", ""),
        "closing_paragraph": gpt_data.get("closing_paragraph", ""),
        "supporting_docs": doc_names or [],
    }

    html = _template.render(**context)
    compliance_fn(
        html,
        client_info.get("state"),
        session_id,
        "goodwill",
        ai_client=ai_client,
    )

    safe_name = safe_filename(creditor)
    pdf_path = output_path / f"Goodwill Request - {safe_name}.pdf"
    pdf_fn(html, str(pdf_path))

    with open(output_path / f"{safe_name}_gpt_response.json", "w") as f:
        json.dump(gpt_data, f, indent=2)

    if audit and audit.level is AuditLevel.VERBOSE:
        audit.log_step(
            "goodwill_letter_generated",
            {"creditor": creditor, "output_pdf": str(pdf_path), "response": gpt_data},
        )


__all__ = ["render_goodwill_letter", "load_creditor_address_map"]
