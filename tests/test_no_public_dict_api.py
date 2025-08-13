import json
from pathlib import Path

# Modules that are allowed to expose raw dicts.
# These are explicit edge adapters such as the HTTP app or CLI layers.
ALLOWLIST_PREFIXES = ("app.py", "app/", "cli/")


def test_no_public_dict_api() -> None:
    inventory = Path(__file__).resolve().parent.parent / "dict_api_inventory.json"
    data = json.loads(inventory.read_text())
    offenders = [
        entry
        for entry in data
        if not any(entry["module"].startswith(prefix) for prefix in ALLOWLIST_PREFIXES)
    ]
    assert (
        not offenders
    ), "Public APIs should use typed models instead of dicts: " + ", ".join(
        f"{o['module']}::{o['function']}" for o in offenders
    )
