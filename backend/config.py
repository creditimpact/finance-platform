import json
import os
from pathlib import Path
from typing import Any, Tuple


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def _load_keyword_lists() -> Tuple[dict, dict, dict]:
    """Load tiered keyword dictionaries from JSON/YAML config.

    Returns empty dictionaries when the config file is missing or malformed.
    """

    path = os.getenv("KEYWORDS_CONFIG_PATH", Path(__file__).with_name("keywords.json"))
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Any = json.load(f)
    except Exception:
        data = {}
    t1 = data.get("tier1") or data.get("TIER1_KEYWORDS") or {}
    t2 = data.get("tier2") or data.get("TIER2_KEYWORDS") or {}
    t3 = data.get("tier3") or data.get("TIER3_KEYWORDS") or {}
    return t1, t2, t3


UTILIZATION_PROBLEM_THRESHOLD = float(
    os.getenv("UTILIZATION_PROBLEM_THRESHOLD", "0.90")
)
SERIOUS_DELINQUENCY_MIN_DPD = int(os.getenv("SERIOUS_DELINQUENCY_MIN_DPD", "60"))

ENABLE_TIER1_KEYWORDS = _env_bool("ENABLE_TIER1_KEYWORDS", False)
ENABLE_TIER2_KEYWORDS = _env_bool("ENABLE_TIER2_KEYWORDS", False)
ENABLE_TIER3_KEYWORDS = _env_bool("ENABLE_TIER3_KEYWORDS", False)
ENABLE_TIER2_NUMERIC = _env_bool("ENABLE_TIER2_NUMERIC", True)

_raw_t1, _raw_t2, _raw_t3 = _load_keyword_lists()

TIER1_KEYWORDS = _raw_t1 if ENABLE_TIER1_KEYWORDS else {}
TIER2_KEYWORDS = _raw_t2 if ENABLE_TIER2_KEYWORDS else {}
TIER3_KEYWORDS = _raw_t3 if ENABLE_TIER3_KEYWORDS else {}
