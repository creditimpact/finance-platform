"""Utilities for applying tolerance checks to numeric amounts."""
from __future__ import annotations

from math import isfinite, isnan
from typing import Iterable, Optional, Tuple


def _to_float(value: object) -> Optional[float]:
    """Attempt to coerce ``value`` to a float, returning ``None`` on failure."""
    if value is None:
        return None

    try:
        coerced = float(value)
    except (TypeError, ValueError):
        return None

    if isnan(coerced) or not isfinite(coerced):
        return None

    return coerced


def are_amounts_within_tolerance(
    values: Iterable[object],
    abs_tol: float,
    ratio_tol: float,
) -> Tuple[bool, Optional[float]]:
    """
    Determine whether the numeric values provided stay within the configured tolerance.

    Parameters
    ----------
    values:
        Iterable of raw values which may be numeric or coercible to float.
    abs_tol:
        Absolute dollar tolerance. Differences below this value are ignored.
    ratio_tol:
        Ratio tolerance expressed as a decimal (e.g., ``0.01`` for 1%).

    Returns
    -------
    Tuple[bool, Optional[float]]
        ``True``/``False`` for whether the values are within tolerance, and the
        computed absolute difference if at least two numeric values were provided.
    """

    numeric_values = [value for value in (_to_float(v) for v in values) if value is not None]

    if len(numeric_values) < 2:
        return True, None

    maximum = max(numeric_values)
    minimum = min(numeric_values)
    diff = maximum - minimum

    ratio_cap = abs(maximum) * ratio_tol
    threshold = max(abs_tol, ratio_cap)

    return diff <= threshold, diff
