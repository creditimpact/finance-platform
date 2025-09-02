from __future__ import annotations

from typing import Any, Dict, Mapping


def is_filled(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and v.strip() == "":
        return False
    if isinstance(v, (list, dict)) and len(v) == 0:
        return False
    return True


def list_merge_preserves(old: list[Any], new: list[Any], key: str | None = None) -> None:
    if key:
        new_index = {item[key]: item for item in new if isinstance(item, Mapping) and key in item}
        for item in old:
            assert isinstance(item, Mapping) and key in item, f"Missing key {key} in old item"
            ident = item[key]
            assert ident in new_index, f"Missing item with {key}={ident}"
            new_item = new_index[ident]
            if isinstance(item, Mapping) and isinstance(new_item, Mapping):
                dict_superset(new_item, item)
    else:
        for item in old:
            assert item in new, f"Missing list item {item}"


def dict_superset(
    new: Mapping[str, Any],
    old: Mapping[str, Any],
    key_map: Dict[str, str] | None = None,
) -> None:
    key_map = key_map or {}
    for k, old_val in old.items():
        assert k in new, f"Missing key {k}"
        new_val = new[k]
        if isinstance(old_val, Mapping) and isinstance(new_val, Mapping):
            dict_superset(new_val, old_val, key_map)
        elif isinstance(old_val, list) and isinstance(new_val, list):
            list_merge_preserves(old_val, new_val, key_map.get(k))
        else:
            if is_filled(old_val):
                assert is_filled(new_val), f"Value for {k} became empty"
