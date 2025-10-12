"""Regression tests for the mask shim on ``NormalizedAccountNumber``."""
from __future__ import annotations

import pytest

from backend.core.merge import acctnum


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "****1234",
            {"has_mask": True, "visible_digits": 4, "canon_mask": "*1234", "digits": "1234"},
        ),
        (
            "XX-XX-4321",
            {"has_mask": True, "visible_digits": 4, "canon_mask": "*4321", "digits": "4321"},
        ),
        (
            "*_*-*  6543",
            {"has_mask": True, "visible_digits": 4, "canon_mask": "*6543", "digits": "6543"},
        ),
    ],
)
def test_masked_displays_collapse_to_star_prefixed_last4(raw: str, expected: dict[str, object]) -> None:
    """Mask characters of any flavor collapse to ``*`` while retaining trailing digits."""

    normalized = acctnum.normalize_display(raw)
    assert normalized.digits == expected["digits"]
    assert normalized.has_mask is expected["has_mask"]
    assert normalized.visible_digits == expected["visible_digits"]
    assert normalized.canon_mask == expected["canon_mask"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "ending in 1234",
            {
                "has_mask": False,
                "visible_digits": 4,
                "canon_mask": "1234",
                "digits": "1234",
            },
        ),
        (
            "123456",
            {
                "has_mask": False,
                "visible_digits": 6,
                "canon_mask": "123456",
                "digits": "123456",
            },
        ),
        (
            "N/A",
            {
                "has_mask": False,
                "visible_digits": 0,
                "canon_mask": "N/A",
                "digits": "",
            },
        ),
        (
            "—",
            {
                "has_mask": False,
                "visible_digits": 0,
                "canon_mask": "",
                "digits": "",
            },
        ),
        (
            "",
            {
                "has_mask": False,
                "visible_digits": 0,
                "canon_mask": "",
                "digits": "",
            },
        ),
    ],
)
def test_non_masked_text_preserves_digits_without_flagging_mask(
    raw: str, expected: dict[str, object]
) -> None:
    """"Ending in …" phrases keep digits but are not masked; plain text is echoed."""

    normalized = acctnum.normalize_display(raw)
    assert normalized.digits == expected["digits"]
    assert normalized.has_mask is expected["has_mask"]
    assert normalized.visible_digits == expected["visible_digits"]
    assert normalized.canon_mask == expected["canon_mask"]
