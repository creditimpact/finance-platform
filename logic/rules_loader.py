from pathlib import Path
import yaml

RULES_DIR = Path(__file__).resolve().parents[1] / "rules"


def _load_yaml(filename: str):
    path = RULES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required rules file: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in {path}: {exc}") from exc


def load_rules() -> list:
    """Load and return dispute rules from ``dispute_rules.yaml``.

    Returns a list of rule dictionaries.
    """
    data = _load_yaml("dispute_rules.yaml")
    if not isinstance(data, list):
        raise ValueError("dispute_rules.yaml must define a list of rules")
    return data


def load_neutral_phrases() -> dict:
    """Load and return neutral phrases mapping from ``neutral_phrases.yaml``."""
    data = _load_yaml("neutral_phrases.yaml")
    if not isinstance(data, dict):
        raise ValueError("neutral_phrases.yaml must define a mapping")
    return data


def load_state_rules() -> dict:
    """Load and return state compliance rules from ``state_rules.yaml``."""
    data = _load_yaml("state_rules.yaml")
    if not isinstance(data, dict):
        raise ValueError("state_rules.yaml must define a mapping")
    return data
