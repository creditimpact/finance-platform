"""Rendering utilities for dispute letters."""

from __future__ import annotations

from backend.assets.paths import templates_path
from backend.core.logic.rendering import pdf_renderer
from backend.core.models.letter import LetterArtifact, LetterContext
from backend.analytics.analytics_tracker import emit_counter


def render_dispute_letter_html(
    context: LetterContext, template_path: str
) -> LetterArtifact:
    """Render the dispute letter HTML using the Jinja template."""

    if not template_path:
        emit_counter("rendering.missing_template_path")
        raise ValueError("template_path is required")

    env = pdf_renderer.ensure_template_env(templates_path(""))
    template = env.get_template(template_path)
    html = template.render(**context.to_dict())
    return LetterArtifact(html=html)


__all__ = ["render_dispute_letter_html"]
