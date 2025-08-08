from __future__ import annotations

from pathlib import Path
from typing import Optional

import pdfkit
from jinja2 import Environment, FileSystemLoader

from config import get_app_config

_template_env: Environment | None = None


def ensure_template_env(base_template_dir: Optional[str] = None) -> Environment:
    """Return a Jinja2 environment rooted at ``base_template_dir``.

    The environment is cached so multiple calls reuse the same loader.
    """
    global _template_env
    base_dir = base_template_dir or "templates"
    if _template_env is None or getattr(_template_env.loader, "searchpath", [None])[0] != base_dir:
        _template_env = Environment(loader=FileSystemLoader(base_dir))
    return _template_env


def normalize_output_path(path: str) -> str:
    """Normalize an output path for PDF generation.

    Ensures the path is absolute, ends with ``.pdf`` and that the parent
    directory exists.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if p.suffix.lower() != ".pdf":
        p = p.with_suffix(".pdf")
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p.resolve())


def render_html_to_pdf(
    html: str,
    output_path: str,
    *,
    wkhtmltopdf_path: Optional[str] = None,
) -> None:
    """Render ``html`` to a PDF at ``output_path``.

    Parameters
    ----------
    html:
        The HTML string to convert.
    output_path:
        Desired file path for the resulting PDF. The path is normalized using
        :func:`normalize_output_path`.
    wkhtmltopdf_path:
        Optional path to the ``wkhtmltopdf`` executable. Defaults to the
        repository-wide configuration.
    """
    output_path = normalize_output_path(output_path)
    wkhtmltopdf = wkhtmltopdf_path or get_app_config().wkhtmltopdf_path
    try:
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf)
        options = {"quiet": ""}
        pdfkit.from_string(html, output_path, configuration=config, options=options)
        print(f"[📄] PDF rendered: {output_path}")
    except Exception as e:  # pragma: no cover - rendering failures are logged
        print(f"[❌] Failed to render PDF: {e}")
