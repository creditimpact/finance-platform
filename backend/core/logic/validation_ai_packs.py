"""Helpers for building validation AI adjudication packs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

log = logging.getLogger(__name__)


def _normalize_indices(indices: Iterable[int | str]) -> list[int]:
    normalized: set[int] = set()
    for idx in indices:
        try:
            normalized.add(int(str(idx)))
        except Exception:
            continue
    return sorted(normalized)


def build_validation_ai_packs_for_accounts(
    sid: str,
    *,
    account_indices: Sequence[int | str],
    runs_root: Path | str | None = None,
) -> None:
    """Trigger validation AI pack building for the provided account indices.

    This is a lightweight shim that simply logs the request for now. The actual
    pack construction logic will be implemented in subsequent iterations.
    """

    normalized_indices = _normalize_indices(account_indices)
    if not normalized_indices:
        return

    log.info(
        "VALIDATION_AI_PACKS_TRIGGER sid=%s runs_root=%s accounts=%s",
        sid,
        str(runs_root) if runs_root is not None else None,
        ",".join(str(idx) for idx in normalized_indices),
    )

