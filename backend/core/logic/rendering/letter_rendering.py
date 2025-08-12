"""Rendering utilities for dispute letters."""

from __future__ import annotations

from backend.core.logic.rendering import pdf_renderer
from backend.core.models.letter import LetterContext, LetterArtifact
from backend.assets.paths import templates_path


def render_dispute_letter_html(context: LetterContext) -> LetterArtifact:
    """Render the dispute letter HTML using the Jinja template."""

    env = pdf_renderer.ensure_template_env(templates_path(""))
    template = env.get_template("dispute_letter_template.html")
    html = template.render(**context.to_dict())
    return LetterArtifact(html=html)


__all__ = ["render_dispute_letter_html"]
