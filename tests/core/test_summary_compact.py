from __future__ import annotations

from backend.core.logic.summary_compact import compact_merge_sections


def test_compact_merge_sections_filters_and_normalizes() -> None:
    summary = {
        "merge_scoring": {
            "best_with": "2",
            "score_total": 42.0,
            "reasons": ("foo", "bar"),
            "conflicts": {"baz", "qux"},
            "identity_score": "7",
            "debt_score": None,
            "acctnum_level": "strong",
            "matched_fields": {"name": 1, "ssn": "yes", "phone": 0},
            "acctnum_digits_len_a": "4",
            "acctnum_digits_len_b": 5,
            "extra": "remove me",
        },
        "merge_explanations": [
            {
                "kind": "merge_pair",
                "with": "3",
                "decision": "ai",
                "total": 10.5,
                "parts": {"match": "8", "conflict": 1.2, "noop": None},
                "matched_fields": {"name": "true", "phone": []},
                "reasons": ["keep"],
                "conflicts": ("drop",),
                "strong": 0,
                "acctnum_level": "medium",
                "acctnum_digits_len_a": "6",
                "acctnum_digits_len_b": "7",
                "tiebreaker": "remove",
            },
            "not a mapping",
        ],
        "untouched": {"stay": True},
    }

    compact_merge_sections(summary)

    assert summary["merge_scoring"] == {
        "best_with": 2,
        "score_total": 42,
        "reasons": ["foo", "bar"],
        "conflicts": ["baz", "qux"],
        "identity_score": 7,
        "acctnum_level": "strong",
        "matched_fields": {"name": True, "ssn": True, "phone": False},
        "acctnum_digits_len_a": 4,
        "acctnum_digits_len_b": 5,
    }

    assert summary["merge_explanations"] == [
        {
            "kind": "merge_pair",
            "with": 3,
            "decision": "ai",
            "total": 10,
            "parts": {"match": 8, "conflict": 1},
            "matched_fields": {"name": True, "phone": False},
            "reasons": ["keep"],
            "conflicts": ["drop"],
            "strong": False,
            "acctnum_level": "medium",
            "acctnum_digits_len_a": 6,
            "acctnum_digits_len_b": 7,
        }
    ]

    assert summary["untouched"] == {"stay": True}
