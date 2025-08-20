from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("template_config.yaml")


def validate() -> None:
    with CONFIG_PATH.open() as f:
        data = yaml.safe_load(f) or {}

    errors = []
    for tag, templates in data.items():
        if not templates:
            errors.append(f"action_tag '{tag}' has no candidate templates")
            continue
        for entry in templates:
            if "required_fields" not in entry or not entry["required_fields"]:
                tpl = entry.get("template", "<unknown>")
                errors.append(
                    f"template '{tpl}' missing required_fields for action_tag '{tag}'"
                )

    if errors:
        raise SystemExit("Config validation failed:\n" + "\n".join(errors))


if __name__ == "__main__":
    validate()
