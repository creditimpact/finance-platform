"""Rendering utilities for dispute letters."""

from __future__ import annotations

from pathlib import Path

from logic import pdf_renderer
from models.letter import LetterContext, LetterArtifact, LetterAccount
from models.account import Inquiry
import warnings


def render_dispute_letter_html(context: LetterContext | dict) -> LetterArtifact:
    """Render the dispute letter HTML using the Jinja template.

    Accepts either a :class:`LetterContext` or a legacy ``dict``. When a dict
    is supplied a :class:`DeprecationWarning` is emitted and the input is
    normalized to ``LetterContext``.
    """

    if isinstance(context, dict):
        try:
            warnings.warn(
                "dict context is deprecated", DeprecationWarning, stacklevel=2
            )
        except DeprecationWarning:
            pass
        context = LetterContext.from_dict(context)

    env = pdf_renderer.ensure_template_env("templates")
    template = env.get_template("dispute_letter_template.html")
    html = template.render(**context.to_dict())
    return LetterArtifact(html=html)


def render_html_to_pdf(
    html_string: str, output_path: Path, wkhtmltopdf_path: str | None = None
) -> None:
    """Thin wrapper around :func:`logic.pdf_renderer.render_html_to_pdf`."""

    pdf_renderer.render_html_to_pdf(
        html_string, str(output_path), wkhtmltopdf_path=wkhtmltopdf_path
    )


__all__ = ["render_dispute_letter_html", "render_html_to_pdf"]

