from __future__ import annotations

import functools
import json
import pathlib


@functools.lru_cache(maxsize=1)
def load_seed_templates() -> dict:
    here = pathlib.Path(__file__).resolve().parent
    path = (here / "seed_argument_templates.json").resolve()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("templates", {})


def build_seed_argument(field: str, c_code: str) -> dict | None:
    tpl = load_seed_templates().get(field, {}).get(c_code)
    if not tpl:
        return None
    return {
        "seed": {
            "id": f"{field}__{c_code}",
            "tone": tpl.get("tone", "firm_courteous"),
            "text": tpl["text"].strip()
        }
    }
