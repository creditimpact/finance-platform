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
