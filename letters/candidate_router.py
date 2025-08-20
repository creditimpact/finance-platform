from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from backend.analytics.analytics_tracker import emit_counter
from backend.audit.audit import AuditLogger


TEMPLATES_DIR = Path(__file__).with_name("templates")


def _get_source(evidence: Any) -> str:
    """Return the CRA source for ``evidence`` if available."""

    if evidence is None:
        return ""
    source = getattr(evidence, "source", None)
    if source is None and isinstance(evidence, dict):
        source = evidence.get("source")
    return (source or "").lower()


def select_template(
    action_tag: str,
    bureau: str | None,
    evidence: Any,
    audit: AuditLogger | None = None,
    templates_dir: Path | None = None,
) -> str:
    """Select a template for ``action_tag`` and ``bureau``.

    CRA-specific templates (``*_bureau_*``) are preferred when the evidence
    source matches the bureau. Generic templates are used when no specific
    match is found. If no candidate templates are available the function
    falls back to ``default_dispute.html`` and emits
    ``router.candidate_selected{tag=default}``.

    A deterministic hash of ``(action_tag, template_path)`` is recorded in the
    audit log for tracing.
    """

    templates_dir = templates_dir or TEMPLATES_DIR
    tag = (action_tag or "").lower()
    bureau = (bureau or "").lower()

    candidates = sorted(templates_dir.glob(f"{tag}*.html"), key=lambda p: p.name)

    selected: str | None = None
    evidence_source = _get_source(evidence)

    if evidence_source == bureau:
        for path in candidates:
            if f"_bureau_{bureau}" in path.stem.lower():
                selected = path.name
                break

    if selected is None and candidates:
        for path in candidates:
            if "_bureau_" not in path.name.lower():
                selected = path.name
                break
        if selected is None:
            # all candidates were bureau-specific but none matched
            selected = candidates[0].name

    metric_tag = tag
    if selected is None:
        selected = "default_dispute.html"
        metric_tag = "default"

    emit_counter("router.candidate_selected", {"tag": metric_tag})

    if audit is not None:
        payload = f"{tag}:{selected}".encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        audit.log_step("candidate_router", {"hash": digest})

    return selected


__all__ = ["select_template"]
