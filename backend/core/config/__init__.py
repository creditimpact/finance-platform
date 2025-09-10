"""Core configuration flags.

This module exposes simple boolean switches used across the codebase.  Flags
are defined here rather than scattering ``os.getenv`` calls throughout the
code, which keeps orchestrator logic straightforward and easy to test.
"""

# Cleanup flag ---------------------------------------------------------------
#
# Stage-A export writes numerous intermediate trace files under
# ``traces/blocks/<sid>``.  Once the three final artifacts are produced we can
# optionally purge the rest to save disk space.  The orchestrator consults this
# flag to determine whether the cleanup routine should run.
from __future__ import annotations

# Whether to remove trace files after Stage-A export.  Defaulted ``False`` so
# cleanup occurs in the Celery chain; tests may override via monkeypatching.
CLEANUP_AFTER_EXPORT: bool = False

__all__ = ["CLEANUP_AFTER_EXPORT"]
