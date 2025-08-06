import json
import os
import threading
from typing import Any, Dict

SESSION_FILE = os.path.join("data", "sessions.json")
_lock = threading.Lock()


def _load_sessions() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(SESSION_FILE):
        return {}
    try:
        with open(SESSION_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_sessions(sessions: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(SESSION_FILE), exist_ok=True)
    with open(SESSION_FILE, 'w', encoding='utf-8') as f:
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
        session.update(kwargs)
        sessions[session_id] = session
        _save_sessions(sessions)
        return session
