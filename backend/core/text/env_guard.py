from __future__ import annotations

import os
import shutil
import logging
from typing import Dict, Any


logger = logging.getLogger(__name__)
_checked = False


def ensure_env_and_paths() -> Dict[str, Any]:
    """Lightweight environment self-check for dev convenience.

    - Validates Tesseract availability (warns if missing; no hard exit).
    - Sets defaults for critical toggles if unset (BLOCK_DEBUG, USE_LAYOUT_TEXT).
    - Warns if AUDIT_PDF is missing (useful for the audit CLI).

    Returns a small summary dict. Safe to call multiple times.
    """
    global _checked
    if _checked:
        return {}
    _checked = True

    summary: Dict[str, Any] = {}

    # Critical toggles: set sane defaults if not explicitly provided
    if os.getenv("BLOCK_DEBUG") is None:
        os.environ["BLOCK_DEBUG"] = "0"
        summary["BLOCK_DEBUG"] = "0"
    if os.getenv("USE_LAYOUT_TEXT") is None:
        os.environ["USE_LAYOUT_TEXT"] = "1"
        summary["USE_LAYOUT_TEXT"] = "1"

    # Tesseract availability
    cmd = os.getenv("TESSERACT_CMD") or "tesseract"
    path = shutil.which(cmd) or shutil.which("tesseract")
    if not path:
        logger.warning("ENV_GUARD: tesseract not found; OCR layout may be unavailable")
        summary["tesseract"] = "missing"
    else:
        # Keep existing TESSERACT_CMD if set; otherwise set to discovered path
        if os.getenv("TESSERACT_CMD") is None:
            os.environ["TESSERACT_CMD"] = path
        summary["tesseract"] = path

    # Audit helper hint
    if not os.getenv("AUDIT_PDF"):
        logger.info("ENV_GUARD: AUDIT_PDF not set (scripts/audit_tp_layout_pipeline.py)")

    return summary


__all__ = ["ensure_env_and_paths"]

