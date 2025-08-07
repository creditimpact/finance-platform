"""Rendering utilities for dispute letters."""

from __future__ import annotations

from pathlib import Path

from config import WKHTMLTOPDF_PATH
from logic import pdf_renderer


def render_dispute_letter_html(context: dict) -> str:
    """Render the dispute letter HTML using the Jinja template."""

    env = pdf_renderer.ensure_template_env("templates")
    template = env.get_template("dispute_letter_template.html")
    return template.render(**context)


def render_html_to_pdf(html_string: str, output_path: Path) -> None:
    """Thin wrapper around :func:`logic.pdf_renderer.render_html_to_pdf`."""

    pdf_renderer.render_html_to_pdf(html_string, str(output_path))


__all__ = ["render_dispute_letter_html", "render_html_to_pdf", "WKHTMLTOPDF_PATH"]

