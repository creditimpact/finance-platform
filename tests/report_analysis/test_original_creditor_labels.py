from __future__ import annotations

from backend.core.logic.report_analysis import block_exporter
from backend.core.logic.report_analysis import canonical_labels


def test_original_creditor_field_labels_cover_variants() -> None:
    expected_key = "original_creditor"
    label_variants = {
        "original creditor",
        "original creditor 01",
        "original creditor 02",
        "orig. creditor",
        "orig creditor",
    }

    for label in label_variants:
        assert (
            block_exporter.FIELD_LABELS.get(label) == expected_key
        ), f"missing {label!r} in FIELD_LABELS"


def test_original_creditor_regex_map_matches_variants() -> None:
    expected_key = "original_creditor"
    regex_labels = [
        (pattern, key)
        for pattern, key in block_exporter.LABEL_MAP
        if key == expected_key
    ]
    # Sanity: ensure we have at least one regex mapping
    assert regex_labels, "expected an original creditor regex mapping"

    samples = (
        "Original Creditor",
        "Original Creditor 01",
        "Original Creditor 02",
        "Orig. Creditor",
        "Orig Creditor",
    )

    for sample in samples:
        assert any(pat.match(sample) for pat, _ in regex_labels), sample


def test_original_creditor_canonical_labels_cover_variants() -> None:
    expected_key = "original_creditor"
    canonical_map = canonical_labels.LABEL_MAP

    variants = (
        "Original Creditor",
        "Original Creditor 01",
        "Original Creditor 02",
        "Orig. Creditor",
        "Orig Creditor",
    )

    for variant in variants:
        assert (
            canonical_map.get(variant) == expected_key
        ), f"missing {variant!r} in canonical LABEL_MAP"


def test_original_creditor_canonical_schema_includes_variants() -> None:
    canonical_schema = canonical_labels.LABEL_SCHEMA

    schema_variants = (
        "Original Creditor:",
        "Original Creditor 01:",
        "Original Creditor 02:",
        "Orig. Creditor:",
        "Orig Creditor:",
    )

    for variant in schema_variants:
        assert (
            canonical_schema.get(variant) == "text"
        ), f"missing {variant!r} in canonical LABEL_SCHEMA"
