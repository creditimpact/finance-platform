from backend.frontend.packs import generator


def test_derive_masked_display_prefers_existing_display() -> None:
    payload = {"display": "XX1234", "last4": "1234"}

    assert generator._derive_masked_display(payload) == "XX1234"


def test_derive_masked_display_masks_from_last4_when_missing_display() -> None:
    payload = {"last4": "9876"}

    assert generator._derive_masked_display(payload) == "****9876"


def test_derive_masked_display_falls_back_to_minimal_mask() -> None:
    assert generator._derive_masked_display(None) == "****"
    assert generator._derive_masked_display({}) == "****"
