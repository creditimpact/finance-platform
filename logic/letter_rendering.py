"""Rendering utilities for dispute letters."""

from __future__ import annotations

import os
from pathlib import Path

import pdfkit
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader


load_dotenv()
WKHTMLTOPDF_PATH = os.getenv("WKHTMLTOPDF_PATH", "wkhtmltopdf")


def render_dispute_letter_html(context: dict) -> str:
    """Render the dispute letter HTML using the Jinja template."""

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("dispute_letter_template.html")
    return template.render(**context)


def render_html_to_pdf(html_string: str, output_path: Path) -> None:
    """Convert an HTML string to a PDF file at the given path."""

    try:
        config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)
        options = {"quiet": ""}
        pdfkit.from_string(html_string, str(output_path), configuration=config, options=options)
        print(f"[📄] PDF rendered: {output_path}")
    except Exception as e:  # pragma: no cover - rendering failures are logged
        print(f"[❌] Failed to render PDF: {e}")


__all__ = ["render_dispute_letter_html", "render_html_to_pdf", "WKHTMLTOPDF_PATH"]

