import pytest

from backend.util.json_tools import try_fix_to_json


def test_try_fix_to_json_raises_for_none():
    with pytest.raises(ValueError):
        try_fix_to_json(None)


def test_try_fix_to_json_extracts_fenced_block():
    text = "Noise```json\n{\"ok\": true}\n```more"
    assert try_fix_to_json(text) == {"ok": True}


def test_try_fix_to_json_rejects_non_object():
    assert try_fix_to_json("[1, 2, 3]") is None


def test_try_fix_to_json_ignores_blank_strings():
    assert try_fix_to_json("   \n\t  ") is None
