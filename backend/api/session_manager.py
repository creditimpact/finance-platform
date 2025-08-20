import json
import os
import threading
from typing import Any, Dict

from backend.assets.paths import data_path
from backend.core.logic.strategy.summary_classifier import invalidate_summary_cache

# Standard session data that can be safely accessed by most modules.
SESSION_FILE = data_path("sessions.json")
_lock = threading.Lock()

# Dedicated storage for intake-only information such as the raw client
# explanations. These values should never be exposed to downstream
# components like the letter generator.
INTAKE_FILE = data_path("intake_only.json")
_intake_lock = threading.Lock()


def _load_sessions() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(SESSION_FILE):
        return {}
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_sessions(sessions: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f)


def set_session(session_id: str, data: Dict[str, Any]) -> None:
    with _lock:
        sessions = _load_sessions()
        sessions[session_id] = data
        _save_sessions(sessions)


def get_session(session_id: str) -> Dict[str, Any] | None:
    with _lock:
        sessions = _load_sessions()
        return sessions.get(session_id)


def update_session(session_id: str, **kwargs: Any) -> Dict[str, Any]:
    with _lock:
        sessions = _load_sessions()
        session = sessions.get(session_id, {})
        if "structured_summaries" in kwargs:
            invalidate_summary_cache(session_id)
        tri_merge_update = kwargs.pop("tri_merge", None)
        if tri_merge_update:
            tri_merge_session = session.get("tri_merge", {})
            evidence_update = tri_merge_update.pop("evidence", None)
            if evidence_update:
                evidence_session = tri_merge_session.get("evidence", {})
                evidence_session.update(evidence_update)
                tri_merge_session["evidence"] = evidence_session
            tri_merge_session.update(tri_merge_update)
            session["tri_merge"] = tri_merge_session
        session.update(kwargs)
        sessions[session_id] = session
        _save_sessions(sessions)
        return session


# ---------------------------------------------------------------------------
# Intake-only storage helpers
# ---------------------------------------------------------------------------


def _load_intake() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(INTAKE_FILE):
        return {}
    try:
        with open(INTAKE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_intake(intake: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(INTAKE_FILE), exist_ok=True)
    with open(INTAKE_FILE, "w", encoding="utf-8") as f:
        json.dump(intake, f)


def update_intake(session_id: str, **kwargs: Any) -> Dict[str, Any]:
    """Update or create intake-only data for a session.

    This storage is isolated from the main session data and should only be
    accessed by trusted intake components (e.g. the explanations endpoint).
    """

    with _intake_lock:
        intake = _load_intake()
        session = intake.get(session_id, {})
        session.update(kwargs)
        intake[session_id] = session
        _save_intake(intake)
        return session


def get_intake(session_id: str) -> Dict[str, Any] | None:
    """Retrieve intake-only data for a session."""

    with _intake_lock:
        intake = _load_intake()
        return intake.get(session_id)
