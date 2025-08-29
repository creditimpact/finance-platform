"""Helper functions for resolving asset paths.

This module centralizes path construction for assets bundled with the
application.  Using these helpers avoids hard-coding relative paths
throughout the codebase.
"""

from pathlib import Path
import os

ASSETS_ROOT = Path(__file__).parent


def templates_path(name: str) -> str:
    """Return the absolute path to a template asset."""
    return str(ASSETS_ROOT / "templates" / name)


def data_path(name: str) -> str:
    """Return the absolute path to a data asset."""
    return str(ASSETS_ROOT / "data" / name)


def fonts_path(name: str) -> str:
    """Return the absolute path to a font asset."""
    return str(ASSETS_ROOT / "fonts" / name)


def static_path(name: str) -> str:
    """Return the absolute path to a static asset."""
    return str(ASSETS_ROOT / "static" / name)


# --- New helpers for materialized artifacts ---------------------------------


def sessions_path(session_id: str) -> str:
    """Return path to the compact session summary JSON."""
    return str(Path("sessions") / f"{session_id}.json")


def traces_accounts_full_dir(session_id: str) -> str:
    """Return directory for full per-account JSONs for a session."""
    return str(Path("traces") / session_id / "accounts_full")


def account_full_path(session_id: str, account_id: str, slug: str) -> str:
    """Return path to a full per-account JSON file."""
    base = traces_accounts_full_dir(session_id)
    return str(Path(base) / f"{account_id}-{slug}.json")
